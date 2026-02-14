# ============================================================================
# MAIN FLASK APPLICATION - ABSA PLN MOBILE
# ============================================================================

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import plotly
import plotly.graph_objs as go
import plotly.express as px
from werkzeug.utils import secure_filename

# ============================================================================
# KONFIGURASI FLASK APP
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# LOAD MODEL DAN KOMPONEN
# ============================================================================

print("="*80)
print("LOADING MODELS AND COMPONENTS")
print("="*80)

try:
    # Load TF-IDF Vectorizer
    vectorizer = joblib.load('models/best_tfidf_vectorizer.pkl')
    print("✅ TF-IDF Vectorizer loaded")
    
    # Load Models (list of models untuk setiap label)
    models = joblib.load('models/best_models.pkl')
    print(f"✅ Models loaded ({len(models)} label models)")
    
    # Load MultiLabelBinarizer
    mlb = joblib.load('models/best_mlb.pkl')
    print(f"✅ MultiLabelBinarizer loaded")
    print(f"   Label classes: {mlb.classes_}")
    
    # Load Model Info
    with open('models/best_model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"✅ Model info loaded")
    print(f"   Topic Model: {model_info['topic_model']}")
    print(f"   Classifier: {model_info['classifier']}")
    print(f"   F1-Score: {model_info['metrics']['f1_macro']:.4f}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

print("="*80 + "\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_text(text):
    """Preprocessing teks sederhana"""
    text = text.lower()
    return text


def predict_single(text):
    """Prediksi untuk satu teks ulasan"""
    # Preprocessing
    processed_text = preprocess_text(text)
    
    # TF-IDF Transform
    text_tfidf = vectorizer.transform([processed_text])
    
    # Prediksi untuk setiap label
    predictions = np.zeros(len(models))
    probabilities = []
    
    for i, model in enumerate(models):
        pred = model.predict(text_tfidf)[0]
        prob = model.predict_proba(text_tfidf)[0] if hasattr(model, 'predict_proba') else None
        predictions[i] = pred
        probabilities.append(prob)
    
    # Decode predictions
    predicted_labels = mlb.inverse_transform(predictions.reshape(1, -1))[0]
    
    # Parse hasil
    results = {
        'raw_text': text,
        'processed_text': processed_text,
        'predictions': {}
    }
    
    # Daftar semua aspek
    all_aspects = ['User Experience', 'Service Quality']
    
    # Mapping untuk menangani berbagai format nama aspek
    aspect_mapping = {
        'user experience': 'User Experience',
        'userexperience': 'User Experience',
        'user_experience': 'User Experience',
        'service quality': 'Service Quality',
        'servicequality': 'Service Quality',
        'service_quality': 'Service Quality',
    }
    
    # Extract aspek dan sentimen
    detected_aspects = {}
    for label in predicted_labels:
        parts = label.split('_')
        if len(parts) >= 2:
            aspect_parts = parts[:-1]
            sentiment = parts[-1]
            aspect_raw = '_'.join(aspect_parts).lower()
            
            if aspect_raw in aspect_mapping:
                aspect_formatted = aspect_mapping[aspect_raw]
                sentiment_formatted = sentiment.capitalize()
                detected_aspects[aspect_formatted] = sentiment_formatted
            else:
                aspect_formatted = '_'.join(aspect_parts).replace('_', ' ').title()
                if aspect_formatted in all_aspects:
                    sentiment_formatted = sentiment.capitalize()
                    detected_aspects[aspect_formatted] = sentiment_formatted
    
    # Isi semua aspek
    for aspect in all_aspects:
        if aspect in detected_aspects:
            results['predictions'][aspect] = detected_aspects[aspect]
        else:
            results['predictions'][aspect] = '–'
    
    return results


def predict_batch(df):
    """Prediksi untuk batch data (CSV)"""
    results = []
    
    # Cari kolom teks
    text_column = None
    for col in ['ulasan', 'text', 'review', 'comment', 'komentar']:
        if col.lower() in [c.lower() for c in df.columns]:
            text_column = [c for c in df.columns if c.lower() == col.lower()][0]
            break
    
    if text_column is None:
        text_column = df.columns[0]
    
    print(f"✓ Kolom teks: {text_column}")
    
    # Cari kolom tanggal
    date_column = None
    for col in ['tanggal', 'date', 'created_at', 'timestamp']:
        if col.lower() in [c.lower() for c in df.columns]:
            date_column = [c for c in df.columns if c.lower() == col.lower()][0]
            break
    
    # Cari kolom ID
    id_column = None
    for col in ['id', 'ID', 'review_id']:
        if col in df.columns:
            id_column = col
            break
    
    # Prediksi setiap baris
    for idx, row in df.iterrows():
        try:
            text = str(row[text_column])
            
            # Skip baris kosong
            if not text or text.strip() == '' or text.lower() == 'nan':
                continue
            
            prediction = predict_single(text)
            
            result_row = {
                'id': row[id_column] if id_column else (idx + 1),
                'ulasan': text,
            }
            
            if date_column:
                result_row['tanggal'] = row[date_column]
            
            all_aspects = ['User Experience', 'Service Quality']
            for aspect in all_aspects:
                if aspect in prediction['predictions']:
                    result_row[aspect] = prediction['predictions'][aspect]
                else:
                    result_row[aspect] = '–'
            
            results.append(result_row)
            
        except Exception as e:
            print(f"❌ Error di baris {idx}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_visualizations(df_results):
    """Membuat visualisasi untuk hasil batch prediction"""
    graphs = {}
    
    aspect_columns = [col for col in df_results.columns 
                     if col not in ['id', 'ulasan', 'tanggal', 'tanggal_parsed']]
    
    # 1. BAR CHART
    bar_data = []
    for aspect in aspect_columns:
        sentiment_counts = df_results[aspect].value_counts()
        for sentiment, count in sentiment_counts.items():
            if sentiment != '–':
                bar_data.append({
                    'Aspek': aspect,
                    'Sentimen': sentiment,
                    'Jumlah': count
                })
    
    if bar_data:
        df_bar = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            df_bar, x='Aspek', y='Jumlah', color='Sentimen',
            title='Distribusi Sentimen per Aspek',
            labels={'Jumlah': 'Jumlah Ulasan'},
            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'},
            barmode='group'
        )
        fig_bar.update_layout(
            xaxis_title='Aspek', yaxis_title='Jumlah Ulasan',
            font=dict(size=12), showlegend=True,
            plot_bgcolor='white', paper_bgcolor='white'
        )
        graphs['bar_chart'] = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. PIE CHART
    all_sentiments = []
    for aspect in aspect_columns:
        sentiments = df_results[aspect][df_results[aspect] != '–'].tolist()
        all_sentiments.extend(sentiments)
    
    if all_sentiments:
        sentiment_counts = pd.Series(all_sentiments).value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Proporsi Sentimen Keseluruhan',
            color=sentiment_counts.index,
            color_discrete_map={'Positive': "#58ca73", 'Negative': "#e2505e"}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        graphs['pie_chart'] = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. LINE CHARTS
    line_charts = {}
    
    if 'tanggal' in df_results.columns:
        try:
            df_results['tanggal_parsed'] = pd.to_datetime(
                df_results['tanggal'], errors='coerce'
            )
            
            valid_dates = df_results['tanggal_parsed'].notna().sum()
            
            if valid_dates > 0:
                df_sorted = df_results.sort_values('tanggal_parsed')
                
                for aspect in aspect_columns:
                    dates = []
                    pos_counts = []
                    neg_counts = []
                    
                    for date in df_sorted['tanggal_parsed'].dropna().unique():
                        day_data = df_sorted[df_sorted['tanggal_parsed'] == date]
                        sentiments = day_data[aspect][day_data[aspect] != '–']
                        
                        pos_count = (sentiments == 'Positive').sum()
                        neg_count = (sentiments == 'Negative').sum()
                        
                        dates.append(date)
                        pos_counts.append(pos_count)
                        neg_counts.append(neg_count)
                    
                    if dates:
                        fig_line = go.Figure()
                        
                        fig_line.add_trace(go.Scatter(
                            x=dates, y=pos_counts,
                            mode='lines+markers', name='Positif',
                            line=dict(color='green', width=3),
                            marker=dict(size=6)
                        ))
                        
                        fig_line.add_trace(go.Scatter(
                            x=dates, y=neg_counts,
                            mode='lines+markers', name='Negatif',
                            line=dict(color='red', width=3),
                            marker=dict(size=6)
                        ))
                        
                        fig_line.update_layout(
                            title=f'Tren Sentimen: {aspect}',
                            xaxis_title='Tanggal',
                            yaxis_title='Jumlah Ulasan',
                            font=dict(size=12),
                            hovermode='x unified',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=350
                        )
                        
                        line_charts[aspect] = json.dumps(
                            fig_line, cls=plotly.utils.PlotlyJSONEncoder
                        )
            else:
                print("⚠️  Kolom tanggal ada tapi tidak ada data valid")
                
        except Exception as e:
            print(f"⚠️  Error saat membuat line chart: {e}")
            pass
    else:
        print("ℹ️  Kolom tanggal tidak ditemukan, skip line charts")
    
    graphs['line_charts'] = line_charts
    
    return graphs

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Halaman Panduan"""
    return render_template('index.html', model_info=model_info)


@app.route('/analisis-teks')
def analisis_teks():
    """Halaman Analisis Teks Tunggal"""
    return render_template('analisis_teks.html', model_info=model_info)


@app.route('/analisis-csv')
def analisis_csv():
    """Halaman Analisis CSV"""
    return render_template('analisis_csv.html', model_info=model_info)


@app.route('/predict_single', methods=['POST'])
def predict_single_route():
    """Route untuk prediksi teks tunggal"""
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong'}), 400
        
        result = predict_single(text)
        
        return render_template('result_single.html', result=result, model_info=model_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch_route():
    """Route untuk prediksi batch (CSV)"""
    filepath = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File tidak ditemukan'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'File tidak dipilih'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file harus CSV'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'File gagal disimpan'}), 500
        
        print(f"✓ File saved: {filepath}")
        
        # Read CSV
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
        
        if len(df) == 0:
            return jsonify({'error': 'File CSV kosong'}), 400
        
        print(f"✓ CSV loaded: {len(df)} rows")
        
        # Predict
        df_results = predict_batch(df)
        print(f"✓ Prediction done: {len(df_results)} results")
        
        # Create visualizations
        graphs = create_visualizations(df_results)
        
        # ── CREATE TABLE HTML ──────────────────────────────────────
        table_html = '''
<style>
.ulasan-cell {
    max-width: 500px;
    white-space: normal;
    word-wrap: break-word;
}
.ulasan-short { display: block; }
.ulasan-full { display: none; }
.show-more-btn {
    color: #007bff;
    cursor: pointer;
    font-size: 0.85rem;
    text-decoration: underline;
    margin-top: 5px;
    display: inline-block;
}
.show-more-btn:hover { color: #0056b3; }
</style>

<table class="table table-striped table-hover" id="resultTable">
<thead><tr>
'''
        
        # Headers
        for col in df_results.columns:
            if col != 'tanggal_parsed':
                table_html += f'<th>{col}</th>'
        table_html += '</tr></thead><tbody>'
        
        # Rows
        for idx, row in df_results.iterrows():
            table_html += '<tr>'
            for col in df_results.columns:
                if col == 'tanggal_parsed':
                    continue
                
                value = row[col]
                
                # Sentiment columns
                if col in ['User Experience', 'Service Quality', 'System Reliability']:
                    if value == 'Positive':
                        cell_html = f'<td><span style="color: #28a745; font-weight: 600;"><i class="fas fa-smile"></i> {value}</span></td>'
                    elif value == 'Negative':
                        cell_html = f'<td><span style="color: #dc3545; font-weight: 600;"><i class="fas fa-frown"></i> {value}</span></td>'
                    elif value == '–':
                        cell_html = f'<td><span style="color: #999;">–</span></td>'
                    else:
                        cell_html = f'<td>{value}</td>'
                
                elif col == 'ulasan':
                    # Ulasan dengan expand/collapse
                    full_text = str(value)
                    
                    if len(full_text) > 150:
                        short_text = full_text[:150]
                        cell_html = f'''
<td class="ulasan-cell">
    <div class="ulasan-container">
        <span class="ulasan-short" id="short-{idx}">{short_text}...</span>
        <span class="ulasan-full" id="full-{idx}">{full_text}</span>
        <br>
        <a class="show-more-btn" onclick="toggleUlasan({idx})">
            <span id="btn-text-{idx}">Lihat Selengkapnya</span>
        </a>
    </div>
</td>
'''
                    else:
                        cell_html = f'<td class="ulasan-cell">{full_text}</td>'
                
                else:
                    cell_html = f'<td>{value}</td>'
                
                table_html += cell_html
            
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        
        # JavaScript
        table_html += '''
<script>
function toggleUlasan(idx) {
    var shortElem = document.getElementById('short-' + idx);
    var fullElem = document.getElementById('full-' + idx);
    var btnText = document.getElementById('btn-text-' + idx);
    
    if (shortElem.style.display !== 'none') {
        shortElem.style.display = 'none';
        fullElem.style.display = 'block';
        btnText.textContent = 'Sembunyikan';
    } else {
        shortElem.style.display = 'block';
        fullElem.style.display = 'none';
        btnText.textContent = 'Lihat Selengkapnya';
    }
}
</script>
'''
        
        # Cleanup
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                print(f"✓ File deleted: {filepath}")
        except Exception as e:
            print(f"⚠️  Warning: Gagal hapus file {filepath}: {e}")
        
        return render_template(
            'result_batch.html',
            table=table_html,
            total_reviews=len(df_results),
            graphs=graphs,
            model_info=model_info,
            df_results=df_results.to_dict('records')
        )
    
    except Exception as e:
        # Cleanup on error
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi (JSON response)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Field "text" diperlukan'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong'}), 400
        
        result = predict_single(text)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', model_info=model_info), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING FLASK APPLICATION - ABSA PLN MOBILE")
    print("="*80)
    print(f"Model: {model_info['topic_model']} + {model_info['classifier']}")
    print(f"F1-Score: {model_info['metrics']['f1_macro']:.4f}")
    print("="*80)
    print("Available Routes:")
    print("  - http://localhost:5000/              (Panduan)")
    print("  - http://localhost:5000/analisis-teks (Analisis Teks)")
    print("  - http://localhost:5000/analisis-csv  (Analisis CSV)")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)