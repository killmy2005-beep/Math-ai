import streamlit as st
from openai import OpenAI
import os, json, logging, tempfile, time, sqlite3
from datetime import datetime
from io import BytesIO
from PIL import Image
import requests
import speech_recognition as sr
import sympy as sp
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import base64
from pathlib import Path

# =========================
# ===== إعداد API Keys =====
# =========================
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
serper_api_key = os.getenv("SERPER_API_KEY") or st.secrets.get("SERPER_API_KEY", "")
openweather_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY", "")

if not api_key:
    st.error("❌ لم يتم العثور على مفتاح OpenAI API. يرجى إضافة المفتاح في إعدادات التطبيق.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# ===== نظام الترجمة =====
# =========================
class TranslationSystem:
    def __init__(self):
        self.translations = {}
        self.current_language = "ar"  # اللغة الافتراضية
        
    def load_translations(self, language_code):
        try:
            # ترجمات افتراضية
            default_translations = {
                "ar": {
                    "welcome": "مرحبًا بك في مساعد الذكاء الاصطناعي المتكامل",
                    "settings": "الإعدادات",
                    "enable_voice": "تمكين الصوت",
                    "enable_vision": "تمكين الرؤية",
                    "enable_web": "تمكين البحث على الويب",
                    "chat_placeholder": "اكتب رسالتك هنا...",
                    "error_no_api_key": "❌ لم يتم العثور على مفتاح OpenAI API",
                    "weather_in": "الطقس في",
                    "temperature": "درجة الحرارة",
                    "humidity": "الرطوبة",
                    "voice_response": "رد صوتي",
                    "translation": "ترجمة",
                    "summary": "ملخص",
                    "current_language": "اللغة الحالية",
                    "math_tab": "الرياضيات",
                    "images_tab": "الصور",
                    "voice_tab": "الصوت",
                    "code_tab": "الأكواد",
                    "web_tab": "ويب",
                    "translate_tab": "ترجمة/تلخيص",
                    "stats_tab": "إحصاءات"
                },
                "en": {
                    "welcome": "Welcome to the Integrated AI Assistant",
                    "settings": "Settings",
                    "enable_voice": "Enable Voice",
                    "enable_vision": "Enable Vision",
                    "enable_web": "Enable Web Search",
                    "chat_placeholder": "Type your message here...",
                    "error_no_api_key": "❌ OpenAI API key not found",
                    "weather_in": "Weather in",
                    "temperature": "Temperature",
                    "humidity": "Humidity",
                    "voice_response": "Voice response",
                    "translation": "Translation",
                    "summary": "Summary",
                    "current_language": "Current language",
                    "math_tab": "Math",
                    "images_tab": "Images",
                    "voice_tab": "Voice",
                    "code_tab": "Code",
                    "web_tab": "Web",
                    "translate_tab": "Translate/Summarize",
                    "stats_tab": "Statistics"
                }
            }
            
            if language_code in default_translations:
                self.translations[language_code] = default_translations[language_code]
            else:
                self.translations[language_code] = default_translations["en"]
        
        except Exception as e:
            st.error(f"Error loading translations: {str(e)}")
            self.translations[language_code] = default_translations["en"]
    
    def set_language(self, language_code):
        if language_code not in self.translations:
            self.load_translations(language_code)
        self.current_language = language_code
    
    def get(self, key, default=None):
        if (self.current_language in self.translations and 
            key in self.translations[self.current_language]):
            return self.translations[self.current_language][key]
        return default if default else key

# دالة مساعدة للترجمة
def t(key, default=None):
    if "translation" not in st.session_state:
        st.session_state.translation = TranslationSystem()
        st.session_state.translation.load_translations("ar")
    return st.session_state.translation.get(key, default)

# =========================
# ===== تسجيل التفاعلات ===
# =========================
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_interaction(role, content, additional_info=""):
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "role": role, "content": content[:200], "additional_info": additional_info}
    logging.info(json.dumps(entry, ensure_ascii=False))
    
    if "db" in st.session_state:
        conn = st.session_state.db
        c = conn.cursor()
        c.execute("INSERT INTO chat_logs (timestamp, role, content, additional_info) VALUES (?, ?, ?, ?)",
                  (timestamp, role, content[:200], additional_info))
        conn.commit()

# =========================
# ===== قاعدة البيانات =====
# =========================
def init_db():
    try:
        conn = sqlite3.connect('chat_app.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_messages
                     (id INTEGER PRIMARY KEY, role TEXT, content TEXT, timestamp TIMESTAMP, sentiment TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_logs
                     (id INTEGER PRIMARY KEY, timestamp TIMESTAMP, role TEXT, content TEXT, additional_info TEXT)''')
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"خطأ في تهيئة قاعدة البيانات: {str(e)}")
        return None

# =========================
# ===== نظام الإضافات =====
# =========================
class PluginSystem:
    def __init__(self):
        self.plugins = {}
    def register_plugin(self, name, func):
        self.plugins[name] = func
    def execute_plugin(self, name, *args, **kwargs):
        if name in self.plugins:
            return self.plugins[name](*args, **kwargs)
        return f"❌ الإضافة '{name}' غير موجود"

# مثال: إضافة حالة الطقس
def weather_plugin(city):
    if not openweather_api_key:
        return t("error_no_api_key")
    try:
        lang_codes = {
            "ar": "ar",
            "en": "en"
        }
        lang_code = lang_codes.get(st.session_state.translation.current_language, "en")
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}&units=metric&lang={lang_code}"
        data = requests.get(url).json()
        if data.get("cod") != 200: 
            return f"❌ لم يتم العثور على مدينة: {city}"
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        hum = data["main"]["humidity"]
        return f"{t('weather_in')} {city}: {desc}, {t('temperature')}: {temp}°م, {t('humidity')}: {hum}%"
    except Exception as e:
        return f"❌ فشل في الحصول على بيانات الطقس: {str(e)}"

# =========================
# ===== تحليل المشاعر =====
# =========================
def analyze_sentiment(text):
    pos=["ممتاز","رائع","جيد","جميل","awesome","great","good","excellent","super"]
    neg=["سيء","غبي","ممل","bad","terrible","horrible","malo","horrible","aburrido"]
    p=sum([1 for w in pos if w in text.lower()])
    n=sum([1 for w in neg if w in text.lower()])
    return "positive" if p>n else "negative" if n>p else "neutral"

def analyze_sentiment_advanced(text):
    try:
        detected_lang = detect_language(text)
        system_messages = {
            "arabic": "قم بتحليل المشاعر في النص التالي وأجب بـ 'positive' أو 'negative' أو 'neutral' فقط.",
            "english": "Analyze the sentiment in the following text and respond with only 'positive', 'negative', or 'neutral'."
        }
        
        system_message = system_messages.get(detected_lang, system_messages["english"])
        
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":system_message},
                {"role":"user","content":text}
            ],
            max_tokens=10, temperature=0.1
        )
        return resp.choices[0].message.content.strip().lower()
    except:
        return analyze_sentiment(text)

# =========================
# ===== تحويل الصوت إلى نص =====
# =========================
def speech_to_text(audio_bytes):
    try:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with sr.AudioFile(tmp_path) as src:
            audio = recognizer.record(src)
        
        os.remove(tmp_path)
        
        lang_codes = {
            "ar": "ar-AR",
            "en": "en-US"
        }
        lang_code = lang_codes.get(st.session_state.translation.current_language, "en-US")
        
        text = recognizer.recognize_google(audio, language=lang_code)
        log_interaction("voice_input", text, "speech_to_text")
        return text
    except sr.UnknownValueError:
        return "❌ لم يتعرف على الكلام في الصوت"
    except sr.RequestError as e:
        return f"❌ خطأ في خدمة التعرف على الصوت: {e}"
    except Exception as e:
        log_interaction("voice_error", str(e), "speech_to_text")
        return f"❌ تعذر التعرف على الصوت: {str(e)}"

# =========================
# ===== استخراج النص من الصور OCR =====
# =========================
def extract_text_from_image(image_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        current_lang = st.session_state.translation.current_language
        lang_map = {
            "ar": ['ar'],
            "en": ['en']
        }
        languages = lang_map.get(current_lang, ['en'])
        
        reader = easyocr.Reader(languages)
        results = reader.readtext(tmp_path)
        os.remove(tmp_path)
        text = " ".join([r[1] for r in results])
        return text if text else "❌ لم يتم العثور على نص في الصورة"
    except Exception as e:
        return f"❌ فشل في استخراج النص: {str(e)}"

# =========================
# ===== البحث على الويب =====
# =========================
def web_search(query, max_results=3):
    if not serper_api_key: 
        return "❌ مفتاح Serper API غير متوفر"
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": max_results})
        headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
        resp = requests.post(url, headers=headers, data=payload).json()
        
        if "organic" not in resp: 
            return "❌ لم يتم العثور على نتائج"
        
        res = []
        for i, r in enumerate(resp["organic"][:max_results], 1):
            res.append(f"{i}. [{r['title']}]({r['link']})\n{r['snippet']}\n")
        
        return "\n".join(res)
    except Exception as e:
        return f"❌ فشل في البحث: {str(e)}"

# =========================
# ===== اكتشاف اللغة =====
# =========================
def detect_language(text):
    if any('\u0600' <= c <= '\u06FF' for c in text): 
        return "arabic"
    return "english"

# =========================
# ===== تلخيص المحادثة =====
# =========================
def summarize_conversation():
    if len(st.session_state.messages) < 5: 
        return "لم تتم محادثة كافية للتلخيص"
    
    conv = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]])
    try:
        detected_lang = detect_language(conv)
        system_messages = {
            "arabic": "قم بتلخيص النقاط الرئيسية في جملتين أو ثلاث.",
            "english": "Summarize the main points in two or three sentences."
        }
        
        system_message = system_messages.get(detected_lang, system_messages["english"])
        
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":system_message},
                      {"role":"user","content":conv}],
            max_tokens=150, temperature=0.3
        )
        st.session_state.session_summary = resp.choices[0].message.content
        return st.session_state.session_summary
    except Exception as e:
        return f"❌ تعذر التلخيص: {str(e)}"

# =========================
# ===== إنشاء PDF =====
# =========================
def export_pdf_bilingual(messages, filename="chat_bilingual.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        content = m['content']
        
        pdf.multi_cell(0, 8, f"{role}: {content}\n")
        pdf.ln(5)
    
    pdf.output(filename)
    log_interaction("system", "PDF exported", f"filename: {filename}")
    return filename

# =========================
# ===== تهيئة حالة التطبيق =====
# =========================
def initialize_app_state():
    if "db" not in st.session_state:
        st.session_state.db = init_db()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = False
    
    if "vision_enabled" not in st.session_state:
        st.session_state.vision_enabled = False
    
    if "web_enabled" not in st.session_state:
        st.session_state.web_enabled = False
    
    if "session_summary" not in st.session_state:
        st.session_state.session_summary = ""
    
    if "translation" not in st.session_state:
        st.session_state.translation = TranslationSystem()
        st.session_state.translation.load_translations("ar")
    
    if "plugin_system" not in st.session_state:
        st.session_state.plugin_system = PluginSystem()
        st.session_state.plugin_system.register_plugin("weather", weather_plugin)

# =========================
# ===== واجهة المستخدم الرئيسية =====
# =========================
def main():
    # تهيئة حالة التطبيق
    initialize_app_state()
    
    # إعداد صفحة الويب
    st.set_page_config(
        page_title=t("welcome"),
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    st.title(t("welcome"))
    
    # الشريط الجانبي
    with st.sidebar:
        st.header(t("settings"))
        
        # اختيار اللغة
        language_options = {
            "العربية": "ar",
            "English": "en"
        }
        selected_language = st.selectbox(
            "🌍 اللغة / Language",
            options=list(language_options.keys()),
            index=0
        )
        st.session_state.translation.set_language(language_options[selected_language])
        
        # الإعدادات
        st.session_state.tts_enabled = st.checkbox(t("enable_voice"))
        st.session_state.vision_enabled = st.checkbox(t("enable_vision"))
        st.session_state.web_enabled = st.checkbox(t("enable_web"))
        
        use_advanced_sentiment = st.checkbox("📊 استخدام تحليل المشاعر المتقدم")
        max_tokens = st.slider("الحد الأقصى للردود", 100, 2000, 500)
        temp = st.slider("درجة الإبداع", 0.0, 1.0, 0.7)
        
        # أدوات إضافية
        if st.button("📄 تنزيل المحادثة PDF"):
            with st.spinner("جاري إنشاء الملف..."):
                file = export_pdf_bilingual(st.session_state.messages)
                with open(file, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="⬇️ تحميل PDF",
                    data=pdf_data,
                    file_name=file,
                    mime="application/pdf"
                )
        
        if st.button("📊 توليد ملخص الجلسة"):
            summary = summarize_conversation()
            st.text_area("ملخص الجلسة", summary, height=150)
        
        st.header("🔌 الإضافات")
        plugin_input = st.text_input("أدخل اسم الإضافة والمعطيات (مثال: weather لندن)")
        if st.button("تشغيل الإضافة") and plugin_input:
            parts = plugin_input.split(" ", 1)
            plugin_name = parts[0]
            plugin_args = parts[1] if len(parts) > 1 else ""
            result = st.session_state.plugin_system.execute_plugin(plugin_name, plugin_args)
            st.info(result)
    
    # التبويبات الرئيسية
    tab_titles = [
        "💬 " + t("chat_placeholder"),
        "🧮 " + t("math_tab"),
        "🖼️ " + t("images_tab"),
        "🎤 " + t("voice_tab"),
        "💻 " + t("code_tab"),
        "🌐 " + t("web_tab"),
        "🌍 " + t("translate_tab"),
        "📊 " + t("stats_tab")
    ]
    
    tabs = st.tabs(tab_titles)
    
    # تبويب المحادثة
    with tabs[0]:
        for msg in st.session_state.messages:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if "sentiment" in msg: 
                    st.caption(f"المشاعر: {msg['sentiment']}")
        
        prompt = st.chat_input(t("chat_placeholder"))
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            log_interaction("user", prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                user_language = detect_language(prompt)
                system_messages = {
                    "arabic": "أنت مساعد ذكي يتحدث العربية",
                    "english": "You are a smart English-speaking assistant"
                }
                
                system_message = system_messages.get(user_language, "You are a smart assistant")
                
                messages_to_send = [{"role": "system", "content": system_message}] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]
                
                try:
                    with st.spinner("جاري التفكير..."):
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages_to_send,
                            max_tokens=max_tokens,
                            temperature=temp
                        )
                    
                    text = resp.choices[0].message.content
                    sentiment = analyze_sentiment_advanced(text) if use_advanced_sentiment else analyze_sentiment(text)
                    
                    # عرض النص مع تأثير الكتابة
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in text.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": text, "sentiment": sentiment})
                    log_interaction("assistant", text, f"sentiment: {sentiment}")
                    
                except Exception as e:
                    st.error(f"❌ تعذر الرد: {str(e)}")
                    log_interaction("error", str(e))
    
    # تبويب الرياضيات
    with tabs[1]:
        st.header("🧮 " + t("math_tab"))
        math_expr = st.text_input("أدخل تعبيراً رياضياً")
        if math_expr:
            try:
                result = sp.sympify(math_expr)
                st.success(f"النتيجة: {result}")
            except:
                st.error("❌ تعبير غير صحيح")
    
    # تبويب الصور
    with tabs[2]:
        st.header("🖼️ " + t("images_tab"))
        uploaded_image = st.file_uploader("رفع صورة", type=["png", "jpg", "jpeg"])
        
        if uploaded_image and st.session_state.vision_enabled:
            img = Image.open(uploaded_image)
            st.image(img, caption="الصورة المرفوعة", use_column_width=True)
            
            with st.spinner("جاري معالجة الصورة..."):
                text = extract_text_from_image(uploaded_image.getvalue())
            
            if "لم يتم العثور" not in text:
                st.info(f"النص المستخرج: {text}")
    
    # تبويب الصوت
    with tabs[3]:
        st.header("🎤 " + t("voice_tab"))
        audio_bytes = st.file_uploader("رفع ملف صوتي", type=["wav", "mp3"])
        
        if audio_bytes:
            with st.spinner("جاري التعرف على الصوت..."):
                text = speech_to_text(audio_bytes.getvalue())
            st.success(f"النص: {text}")
    
    # تبويب الأكواد
    with tabs[4]:
        st.header("💻 " + t("code_tab"))
        code_input = st.text_area("أدخل الكود لتحليله")
        if code_input: 
            st.info("ميزة تحليل الأكواد قيد التطوير")
    
    # تبويب الويب
    with tabs[5]:
        st.header("🌐 " + t("web_tab"))
        web_query = st.text_input("أدخل استعلام البحث")
        if st.button("ابحث على الويب") and web_query:
            with st.spinner("جاري البحث..."):
                results = web_search(web_query)
            st.markdown(results)
    
    # تبويب الترجمة والتلخيص
    with tabs[6]:
        st.header("🌍 " + t("translate_tab"))
        text_to_process = st.text_area("أدخل النص للترجمة أو التلخيص")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(t("translation")):
                try:
                    current_lang = st.session_state.translation.current_language
                    target_lang = "English" if current_lang == "ar" else "Arabic"
                    
                    with st.spinner("جاري الترجمة..."):
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": f"ترجم النص إلى {target_lang}: {text_to_process}"}],
                            max_tokens=500
                        )
                    
                    st.text_area(t("translation"), resp.choices[0].message.content, height=200)
                except Exception as e:
                    st.error(f"❌ فشل في الترجمة: {str(e)}")
        
        with col2:
            if st.button(t("summary")):
                try:
                    with st.spinner("جاري التلخيص..."):
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": f"لخص هذا النص: {text_to_process}"}],
                            max_tokens=300
                        )
                    
                    st.text_area(t("summary"), resp.choices[0].message.content, height=200)
                except Exception as e:
                    st.error(f"❌ فشل في التلخيص: {str(e)}")
    
    # تبويب الإحصاءات
    with tabs[7]:
        st.header("📊 " + t("stats_tab"))
        if st.session_state.messages:
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("عدد الرسائل الكلي", len(st.session_state.messages))
            col2.metric("رسائل المستخدم", len(user_msgs))
            col3.metric("ردود المساعد", len(assistant_msgs))
            
            sentiments = [m.get("sentiment", "neutral") for m in st.session_state.messages if "sentiment" in m]
            if sentiments:
                positive = sentiments.count("positive")
                negative = sentiments.count("negative")
                neutral = sentiments.count("neutral")
                
                fig, ax = plt.subplots()
                ax.pie([positive, negative, neutral], 
                       labels=["إيجابي", "سلبي", "محايد"], 
                       autopct='%1.1f%%', 
                       colors=['green', 'red', 'gray'])
                ax.set_title("توزيع المشاعر")
                st.pyplot(fig)

# تشغيل التطبيق
if __name__ == "__main__":
    main()
