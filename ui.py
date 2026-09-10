"""
app.py
----------------------------------
رابط کاربری (M-03) — نسخه چهارم

اضافه شده نسبت به نسخه قبل:
  - قبل از ساخت چت جدید، یک پنجره‌ی modal (st.dialog) باز می‌شه
  - وارد کردن اسم چت اجباریه؛ دکمه‌ی «ساخت چت» تا وقتی اسم خالیه
    غیرفعال (disabled) می‌مونه
  - انتخاب رنگ از بین ۸ رنگ پاستلی (به شکل دایره‌های رنگی)
  - رنگ انتخابی به صورت یک نقطه‌ی رنگی کنار اسم چت در سایدبار
    نمایش داده می‌شه
"""

import os
import streamlit as st
import chat_database as db

st.set_page_config(page_title="دستیار هوشمند PDF", layout="wide")

db.initialize_database()


# ============================================================
# پالت رنگ‌های پاستلی برای انتخاب چت
# ============================================================
# هر آیتم: (اسم فارسی رنگ برای نمایش/دیباگ، کد هگز)

PASTEL_COLORS = [
    ("صورتی", "#FADADD"),
    ("آبی آسمانی", "#AEC6CF"),
    ("سبز نعنایی", "#B5EAD7"),
    ("زرد کِرِم", "#FFF5BA"),
    ("بنفش کمرنگ", "#C7CEEA"),
    ("هلویی", "#FFDAB9"),
    ("قرمز ملایم", "#FFB3B3"),
    ("فیروزه‌ای", "#B2F7EF"),
]


# ============================================================
# استایل سفارشی
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stButton, .stChatMessage {
        font-family: 'Vazirmatn', sans-serif !important;
    }

    h1 { font-weight: 700 !important; letter-spacing: -0.5px; }

    section[data-testid="stSidebar"] button {
        border-radius: 10px !important;
        text-align: right !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# session_state
# ============================================================

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "uploader_key_counter" not in st.session_state:
    st.session_state.uploader_key_counter = 0

if "new_chat_color" not in st.session_state:
    st.session_state.new_chat_color = PASTEL_COLORS[0][1]


# ============================================================
# پنجره‌ی مودال ساخت چت جدید
# ============================================================
# نکته مهم درباره st.dialog:
# تابعی که با @st.dialog تزئین شده، وقتی "صدا زده بشه" (نه وقتی تعریف
# می‌شه) یه پنجره‌ی modal روی کل صفحه باز می‌کنه. تا وقتی داخل این
# پنجره st.rerun() صدا نزنیم، پنجره باز می‌مونه و بین تعامل‌های مختلف
# (مثل کلیک روی رنگ‌ها) بسته نمی‌شه. st.rerun() تنها جایی صدا زده
# می‌شه که واقعاً می‌خوایم پنجره بسته بشه (یعنی موقع ثبت نهایی چت).

@st.dialog("ساخت چت جدید")
def open_new_chat_dialog():
    chat_name = st.text_input(
        "اسم چت",
        placeholder="مثلاً: پروژه پایان‌نامه",
        key="new_chat_name_input",
    )
    st.caption("این فیلد اجباریه — تا خالیه، دکمه‌ی ساخت غیرفعاله.")

    st.write("رنگ چت را انتخاب کن:")

    cols = st.columns(4)
    for i, (color_name, hex_code) in enumerate(PASTEL_COLORS):
        with cols[i % 4]:
            is_selected = st.session_state.new_chat_color == hex_code
            border = "3px solid #333333" if is_selected else "3px solid transparent"

            st.markdown(
                f"""
                <div style="
                    width:44px; height:44px; border-radius:50%;
                    background-color:{hex_code};
                    border:{border};
                    margin: 4px auto;
                "></div>
                """,
                unsafe_allow_html=True,
            )

            btn_label = "✓" if is_selected else "انتخاب"
            if st.button(btn_label, key=f"color_choice_{i}", use_container_width=True):
                # عمداً st.rerun() صدا نمی‌زنیم؛ خودِ کلیک روی دکمه باعث
                # rerun طبیعی می‌شه و چون پنجره از قبل باز بوده، باز می‌مونه.
                st.session_state.new_chat_color = hex_code

    st.divider()

    name_is_valid = bool(chat_name and chat_name.strip())

    if st.button(
        "✅ ساخت چت",
        use_container_width=True,
        type="primary",
        disabled=not name_is_valid,
    ):
        new_id = db.create_chat(
            title=chat_name.strip(),
            color=st.session_state.new_chat_color,
        )
        st.session_state.active_chat_id = new_id
        st.session_state.uploader_key_counter = 0
        st.session_state.new_chat_color = PASTEL_COLORS[0][1]  # ریست برای دفعه بعد
        st.rerun()  # اینجا پنجره بسته می‌شه و چت جدید نمایش داده می‌شه


# ============================================================
# سایدبار: لیست چت‌ها
# ============================================================

with st.sidebar:
    st.title("💬 چت‌ها")

    if st.button("➕ چت جدید", use_container_width=True):
        open_new_chat_dialog()

    st.divider()

    chats = db.get_chats()
    if not chats:
        st.caption("هنوز چتی نساختی.")

    for chat in chats:
        dot_col, btn_col = st.columns([1, 6])
        with dot_col:
            st.markdown(
                f"""
                <div style="
                    width:14px; height:14px; border-radius:50%;
                    background-color:{chat['color']};
                    margin-top: 10px;
                "></div>
                """,
                unsafe_allow_html=True,
            )
        with btn_col:
            label = chat["title"]
            if chat["id"] == st.session_state.active_chat_id:
                label = f"🟢 {label}"
            if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()


# ============================================================
# بخش اصلی
# ============================================================

st.title("دستیار هوشمند پرسش‌وپاسخ PDF")

if st.session_state.active_chat_id is None:
    st.info("یک چت جدید بساز یا از سایدبار یکی رو انتخاب کن.")
    st.stop()

chat_id = st.session_state.active_chat_id

files_col, chat_col = st.columns([1, 2], gap="medium")


# ------------------------------------------------------------
# باکس ۱: اسناد این چت
# ------------------------------------------------------------

with files_col:
    with st.container(border=True):
        st.subheader("📄 اسناد این چت")

        uploaded = st.file_uploader(
            "فایل‌های PDF را اینجا بارگذاری کنید",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{chat_id}_{st.session_state.uploader_key_counter}",
        )

        if uploaded:
            if st.button("➕ افزودن به این چت", use_container_width=True):
                for uf in uploaded:
                    safe_name = f"chat{chat_id}_{uf.name}"
                    save_path = db.UPLOADS_DIR / safe_name
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())

                    db.add_file(
                        chat_id=chat_id,
                        filename=uf.name,
                        stored_path=str(save_path),
                        status="pending",
                    )

                st.session_state.uploader_key_counter += 1
                st.rerun()

        st.divider()

        files = db.get_files(chat_id)

        if not files:
            st.caption("هنوز فایلی برای این چت اضافه نشده.")
        else:
            status_icons = {
                "pending": "⏳ در انتظار پردازش",
                "processing": "🔄 در حال پردازش",
                "done": "✅ آماده",
                "error": "❌ خطا",
            }

            for f in files:
                row_left, row_right = st.columns([4, 1])
                with row_left:
                    st.write(f"📎 {f['filename']}")
                    st.caption(status_icons.get(f["status"], f["status"]))
                with row_right:
                    if st.button("🗑️", key=f"del_{f['id']}"):
                        stored_path = db.delete_file(f["id"])
                        if stored_path and os.path.exists(stored_path):
                            os.remove(stored_path)
                        st.rerun()


# ------------------------------------------------------------
# باکس ۲: چت
# ------------------------------------------------------------

with chat_col:
    with st.container(border=True, height=480):
        messages = db.get_messages(chat_id)

        if not messages:
            st.caption("سوالت رو از پایین بپرس ⬇")

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


# ============================================================
# ورودی چت
# ============================================================

user_input = st.chat_input("سوالت رو بپرس...")

if user_input:
    db.add_message(chat_id, role="user", content=user_input)

    files = db.get_files(chat_id)

    if not files:
        answer = "لطفاً ابتدا از بخش «اسناد این چت» حداقل یک فایل PDF بارگذاری کنید تا بتونم بر اساس اون بهتون پاسخ بدم."
    else:
        answer = "( اینجا بعداً پاسخ واقعی از مدل زبانی، بر اساس اسناد این چت، میاد )"

    db.add_message(chat_id, role="assistant", content=answer)
    st.rerun()
