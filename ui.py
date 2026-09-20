import os
from nicegui import ui, app
import chat_database as db

# ۱. راه‌اندازی دیتابیس
db.initialize_database()

PASTEL_COLORS = [
    ("صورتی نیون", "#FF75C3"),
    ("آبی مدرن", "#00F5D4"),
    ("سبز زمردی", "#2EC4B6"),
    ("زرد کهکشانی", "#FEE440"),
    ("بنفش برقی", "#7B2CBF"),
    ("مرجانی", "#FF9F1C"),
    ("قرمز سرخابی", "#E63946"),
    ("فیروزه‌ای", "#00BBF9"),
]


# ============================================================
# راهنمای استفاده (دیالوگ مشترک بین همه‌ی صفحات)
# ============================================================
def open_help_dialog():
    """
    راهنمای استفاده از برنامه. تابع جدا شد چون هم توی صفحه‌ی گالری هم
    توی صفحه‌ی چت لازمه، و نمی‌خوایم کدش تکرار بشه.
    """
    with ui.dialog() as dialog, ui.card().classes(
        'bg-slate-900 text-slate-100 w-full max-w-xl border border-slate-800 p-6 rounded-2xl'
    ):
        with ui.row().classes('w-full justify-between items-center mb-2'):
            ui.label('📖 راهنمای استفاده').classes('text-xl font-bold')
            ui.button(icon='close', on_click=dialog.close).props('flat round color=grey size=sm')

        ui.separator().classes('bg-slate-800 mb-2')

        with ui.column().classes('w-full gap-1'):
            with ui.expansion('۱. ساخت گفتگوی جدید', icon='add_circle').classes('w-full text-slate-200'):
                ui.label(
                    'از صفحه‌ی اصلی روی «گفتگوی جدید» بزن، یه اسم و یه رنگ برای '
                    'گفتگو انتخاب کن. هر گفتگو کاملاً مستقل از بقیه‌ست: فایل‌ها و '
                    'پیام‌های هر گفتگو فقط مخصوص همون گفتگو می‌مونن.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۲. بارگذاری اسناد PDF', icon='upload_file').classes('w-full text-slate-200'):
                ui.label(
                    'داخل هر گفتگو، از ستون «اسناد این گفتگو» فایل‌های PDF مورد نظرت '
                    'رو آپلود کن. می‌تونی چند فایل داشته باشی و هر کدوم رو جدا حذف '
                    'کنی. فقط فرمت PDF پذیرفته می‌شه.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۳. پرسیدن سوال', icon='chat').classes('w-full text-slate-200'):
                ui.label(
                    'بعد از بارگذاری حداقل یک فایل، سوالت رو توی کادر پایین بنویس. '
                    'اگر هنوز فایلی بارگذاری نکرده باشی، سیستم بهت یادآوری می‌کنه '
                    'که اول باید سند آپلود کنی.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۴. مشاهده‌ی مرجع پاسخ', icon='find_in_page').classes('w-full text-slate-200'):
                ui.label(
                    'کنار هر پاسخ، شماره‌ی صفحه‌ای که اطلاعات ازش استخراج شده نشون '
                    'داده می‌شه. با کلیک روش می‌تونی همون بخش از سند اصلی رو ببینی.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۵. تغییر نام / حذف گفتگو', icon='edit').classes('w-full text-slate-200'):
                ui.label(
                    'از دکمه‌ی مداد بالای هر گفتگو می‌تونی اسمش رو عوض کنی. برای '
                    'حذف کامل یه گفتگو (همراه با تمام فایل‌ها و پیام‌هاش)، از دکمه‌ی '
                    'زباله‌دان روی کارتش توی صفحه‌ی اصلی استفاده کن.'
                ).classes('text-sm text-slate-400 leading-relaxed')

        with ui.row().classes('w-full justify-end mt-3'):
            ui.button('متوجه شدم', on_click=dialog.close).props('unelevated color=primary')

    dialog.open()


# ============================================================
# صفحه اول: گالری گفتگوها (رووت /)
# ============================================================
@ui.page('/')
def gallery_page():
    ui.query('body').classes('bg-slate-950 text-slate-100 font-sans dir-rtl min-h-screen')

    selected_color = {'hex': PASTEL_COLORS[0][1]}

    # هدر صفحه گالری
    with ui.header().classes('bg-slate-900/80 backdrop-blur-md p-4 justify-between items-center border-b border-slate-800 sticky top-0 z-50'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('space_dashboard', size='32px').classes('text-sky-400')
            ui.label('📚 دستیار هوشمند اسناد').classes('text-2xl font-black text-slate-100')

        # دیالوگ ساخت گفتگوی جدید
        def open_create_modal():
            with ui.dialog() as dialog, ui.card().classes('bg-slate-900 text-slate-100 w-96 border border-slate-800 p-6 rounded-2xl'):
                ui.label('✨ ساخت گفتگوی جدید').classes('text-xl font-bold mb-2')
                title_input = ui.input('نام گفتگو', placeholder='مثلاً: پروژه پردازش تصویر').classes('w-full').props('dark outlined')

                ui.label('انتخاب رنگ گفتگو:').classes('text-sm text-slate-400 mt-3 mb-1')

                @ui.refreshable
                def color_swatches():
                    with ui.row().classes('w-full justify-between mb-4'):
                        for name, hex_code in PASTEL_COLORS:
                            def set_color(h=hex_code):
                                selected_color['hex'] = h
                                color_swatches.refresh()

                            is_selected = selected_color['hex'] == hex_code
                            ring = 'border-2 border-white' if is_selected else 'border-2 border-transparent'
                            ui.element('div').classes(
                                f'w-8 h-8 rounded-full cursor-pointer {ring} '
                                'transition-all hover:scale-110'
                            ).style(f'background-color: {hex_code};').on('click', set_color)

                color_swatches()

                def save():
                    if title_input.value:
                        new_id = db.create_chat(title=title_input.value.strip(), color=selected_color['hex'])
                        dialog.close()
                        ui.navigate.to(f'/chat/{new_id}')

                with ui.row().classes('w-full justify-end gap-2 mt-4'):
                    ui.button('انصراف', on_click=dialog.close).props('flat color=grey')
                    ui.button('ایجاد گفتگو', on_click=save).props('unelevated color=primary')
            dialog.open()

        with ui.row().classes('items-center gap-2'):
            ui.button(icon='help_outline', on_click=open_help_dialog).props('flat round color=white')
            ui.button('➕ گفتگوی جدید', on_click=open_create_modal).props('unelevated color=primary').classes('rounded-xl font-bold')

    # بدنه اصلی گالری
    with ui.column().classes('w-full max-w-7xl mx-auto p-6 gap-6'):

        # کادر جستجو
        search_input = ui.input(placeholder='🔍 جستجو در عنوان گفتگوها...').classes('w-full max-w-md mx-auto').props('dark rounded outlined dense')

        @ui.refreshable
        def render_cards():
            chats = db.get_chats()
            query = (search_input.value or '').strip().lower()

            if query:
                chats = [c for c in chats if query in c['title'].lower()]

            if not chats:
                with ui.column().classes('w-full items-center justify-center py-16'):
                    ui.icon('folder_off', size='64px').classes('text-slate-700 mb-3')
                    ui.label('هیچ گفتگویی یافت نشد.').classes('text-slate-500 text-lg')
            else:
                with ui.grid(columns=3).classes('w-full gap-6'):
                    for chat in chats:
                        c_id = chat['id']
                        with ui.card().classes('bg-slate-900/60 border border-slate-800 hover:border-sky-500/50 transition-all duration-300 p-5 rounded-2xl relative group shadow-lg hover:shadow-sky-500/10'):
                            # نوار نئونی بالا
                            ui.element('div').classes('h-1.5 w-full rounded-full mb-4').style(f'background-color: {chat["color"]}; box-shadow: 0 0 10px {chat["color"]}aa;')

                            ui.label(chat['title']).classes('text-xl font-bold text-slate-100 line-clamp-1 mb-2')

                            with ui.row().classes('items-center gap-2 text-slate-400 text-xs mb-1'):
                                ui.icon('schedule', size='16px')
                                ui.label(f'ایجاد: {chat["created_at"]}')

                            with ui.row().classes('w-full justify-between items-center mt-6 pt-3 border-t border-slate-800/60'):
                                # دکمه حذف چت
                                def confirm_delete(chat_to_del=chat):
                                    with ui.dialog() as d, ui.card().classes('bg-slate-900 border border-slate-800 p-5'):
                                        ui.label(f'آیا از حذف گفتگوی «{chat_to_del["title"]}» مطمئن هستید؟').classes('text-slate-200 font-bold')
                                        with ui.row().classes('w-full justify-end gap-2 mt-4'):
                                            ui.button('انصراف', on_click=d.close).props('flat color=grey')

                                            def do_delete():
                                                # حذف فایل‌های فیزیکی روی دیسک
                                                for f in db.get_files(chat_to_del['id']):
                                                    sp = db.delete_file(f['id'])
                                                    if sp and os.path.exists(sp):
                                                        try:
                                                            os.remove(sp)
                                                        except OSError:
                                                            pass
                                                # نکته: از db.delete_chat() استفاده می‌کنیم، نه SQL
                                                # خام؛ چون این تابع از قبل توی chat_database.py
                                                # تعریف شده و لایه‌ی انتزاعی دیتابیس رو حفظ می‌کنه
                                                # (اگه فردا منطق حذف عوض بشه، فقط یه‌جا لازمه
                                                # اصلاح بشه، نه هرجا که چت حذف می‌شه).
                                                db.delete_chat(chat_to_del['id'])
                                                d.close()
                                                render_cards.refresh()
                                            ui.button('حذف', on_click=do_delete).props('unelevated color=negative')
                                    d.open()

                                ui.button(icon='delete', on_click=confirm_delete).props('flat round color=negative size=sm')
                                ui.button('ورود به گفتگو ➔', on_click=lambda id=c_id: ui.navigate.to(f'/chat/{id}')).props('flat color=primary').classes('font-bold')

        search_input.on('update:model-value', render_cards.refresh)
        render_cards()


# ============================================================
# صفحه دوم: محیط چت و اسناد (رووت /chat/{chat_id})
# ============================================================
@ui.page('/chat/{chat_id}')
def chat_page(chat_id: int):
    ui.query('body').classes('bg-slate-950 text-slate-100 font-sans dir-rtl h-screen overflow-hidden')

    chat = db.get_chat_by_id(chat_id)
    if not chat:
        ui.navigate.to('/')
        return

    # هدر چت
    with ui.header().classes('bg-slate-900/80 backdrop-blur-md p-3 justify-between items-center border-b border-slate-800 h-[64px]'):
        with ui.row().classes('items-center gap-3'):
            ui.button(icon='arrow_forward', on_click=lambda: ui.navigate.to('/')).props('flat round color=white')
            ui.element('div').classes('w-3 h-3 rounded-full').style(f'background-color: {chat["color"]}; box-shadow: 0 0 8px {chat["color"]};')
            ui.label(chat['title']).classes('text-lg font-bold text-slate-100')

        with ui.row().classes('items-center gap-2'):
            ui.button(icon='help_outline', on_click=open_help_dialog).props('flat round color=white size=sm')

            # دیالوگ تغییر نام
            def rename_dialog():
                with ui.dialog() as dialog, ui.card().classes('bg-slate-900 border border-slate-800 p-5 w-80'):
                    ui.label('تغییر نام گفتگو').classes('font-bold text-slate-200 mb-2')
                    t_in = ui.input('نام جدید', value=chat['title']).classes('w-full').props('dark outlined dense')
                    def save():
                        if t_in.value:
                            db.rename_chat(chat_id, t_in.value.strip())
                            dialog.close()
                            ui.navigate.reload()
                    with ui.row().classes('w-full justify-end gap-2 mt-3'):
                        ui.button('انصراف', on_click=dialog.close).props('flat color=grey')
                        ui.button('ذخیره', on_click=save).props('unelevated color=primary')
                dialog.open()

            ui.button(icon='edit', on_click=rename_dialog).props('flat round color=amber size=sm')

    # بدنه دو ستونه
    with ui.row().classes('w-full h-[calc(100vh-64px)] p-4 gap-4 no-wrap'):

        # ۱. ستون مدیریت اسناد (سمت چپ)
        with ui.card().classes('w-1/3 h-full bg-slate-900 border border-slate-800 p-4 flex flex-col rounded-2xl'):
            with ui.row().classes('w-full justify-between items-center mb-3'):
                ui.label('📑 اسناد این گفتگو').classes('text-base font-bold text-slate-200')
                ui.icon('folder', size='20px').classes('text-sky-400')

            # نگه‌داری وضعیت نوتیفیکیشن جاری بین دو رویداد مختلف
            # (شروع آپلود و پایان آپلود) — چون این دو تا callback جدا
            # از هم صدا زده می‌شن، باید یه رفرنس مشترک بینشون داشته باشیم
            upload_status = {'notification': None}

            def handle_begin_upload():
                """
                دقیقاً لحظه‌ای صدا زده می‌شه که آپلود واقعاً شروع شده
                (فایل انتخاب شده و شروع به ارسال کرده). یه نوتیفیکیشن
                با اسپینر نشون می‌دیم که تا پایان کار می‌مونه، تا کاربر
                فکر نکنه چیزی متوقف شده.
                """
                upload_status['notification'] = ui.notification(
                    message='📤 در حال آپلود فایل...',
                    spinner=True,
                    type='ongoing',
                    timeout=None,
                )

            def finish_upload_notification(message: str, success: bool) -> None:
                """پیام نوتیفیکیشن جاری رو به نتیجه‌ی نهایی (موفق/ناموفق) تغییر می‌ده."""
                notification = upload_status['notification']
                if notification is None:
                    ui.notify(message, type='positive' if success else 'negative')
                    return
                notification.spinner = False
                notification.message = message
                notification.type = 'positive' if success else 'negative'
                notification.timeout = 3.0

            async def read_file_bytes(file_obj):
                """
                توی NiceGUI 3.x، خوندن محتوای فایل آپلودی async هست.
                این تابع چند متد رایج رو امتحان می‌کنه تا مستقل از حالت
                دقیق شیء (فایل کوچیک در حافظه / فایل بزرگ روی دیسک) کار
                کنه.
                """
                if hasattr(file_obj, 'read'):
                    result = file_obj.read()
                    if hasattr(result, '__await__'):
                        result = await result
                    return result
                if hasattr(file_obj, 'text'):
                    text = await file_obj.text()
                    return text.encode('utf-8')
                if hasattr(file_obj, '_data'):
                    return file_obj._data
                available = [a for a in dir(file_obj) if not a.startswith('__')]
                raise AttributeError(f"نمی‌تونم محتوای فایل رو بخونم. فیلدهای موجود: {available}")

            # تابع مدیریت آپلود فایل با بررسی پسوند PDF و فایل تکراری
            # نکته: async شد چون طبق مستندات رسمی NiceGUI 3.x، خوندن
            # محتوای فایل آپلودی به‌صورت async انجام می‌شه.
            async def handle_upload(e):
                try:
                    # طبق مستندات رسمی این نسخه، فایل زیرِ e.file هست،
                    # نه مستقیم روی خودِ رویداد (e.name دیگه وجود نداره).
                    file_name = e.file.name

                    # بررسی عدم ورود فایل غیر PDF
                    if not file_name.lower().endswith('.pdf'):
                        finish_upload_notification('❌ فقط فایل‌های با فرمت PDF مجاز هستند!', success=False)
                        return

                    # بررسی فایل تکراری (بر اساس اسم فایل، داخل همین گفتگو)
                    existing_names = {f['filename'] for f in db.get_files(chat_id)}
                    if file_name in existing_names:
                        finish_upload_notification(f'⚠️ «{file_name}» قبلاً به این گفتگو اضافه شده.', success=False)
                        return

                    safe_name = f"chat{chat_id}_{file_name}"
                    save_path = db.UPLOADS_DIR / safe_name

                    content_bytes = await read_file_bytes(e.file)

                    with open(save_path, 'wb') as f:
                        f.write(content_bytes)

                    db.add_file(chat_id=chat_id, filename=file_name, stored_path=str(save_path), status='done')
                    finish_upload_notification(f'✅ سند «{file_name}» با موفقیت اضافه شد', success=True)
                    files_container.refresh()

                except Exception as ex:
                    # نکته مهم: قبلاً اگه خطایی وسط پردازش پیش می‌اومد،
                    # کاملاً بی‌صدا شکست می‌خورد (نه توی UI چیزی نشون
                    # داده می‌شد نه توی دیتابیس چیزی ذخیره می‌شد) و ما
                    # اصلاً نمی‌فهمیدیم چرا. الان متن دقیق خطا رو نشون
                    # می‌دیم تا اگه بازم مشکلی بود، دقیقاً بدونیم کجاست.
                    finish_upload_notification(f'❌ خطا در آپلود: {ex}', success=False)

                finally:
                    # ریست المان آپلود، چه موفق چه ناموفق
                    uploader.reset()
                    upload_status['notification'] = None

            # المان آپلود خودکار و محدود به .pdf
            uploader = ui.upload(
                on_begin_upload=handle_begin_upload,
                on_upload=handle_upload,
                label='انتخاب فایل PDF',
                auto_upload=True
            ).classes('w-full mb-3').props('dark flat bordered accept=".pdf"')

            ui.separator().classes('bg-slate-800 my-2')

            # لیست فایل‌های آپلود شده
            @ui.refreshable
            def files_container():
                files = db.get_files(chat_id)
                if not files:
                    with ui.column().classes('w-full items-center justify-center py-10'):
                        ui.icon('description', size='40px').classes('text-slate-700')
                        ui.label('هنوز هیچ سند PDF آپلود نشده است.').classes('text-slate-500 text-xs mt-2')
                else:
                    with ui.column().classes('w-full flex-grow overflow-y-auto gap-2 pr-1'):
                        for f in files:
                            with ui.row().classes('w-full justify-between items-center bg-slate-800/70 border border-slate-700/50 p-2.5 rounded-xl'):
                                with ui.row().classes('items-center gap-2 overflow-hidden'):
                                    ui.icon('picture_as_pdf', size='20px').classes('text-red-400 flex-shrink-0')
                                    with ui.column().classes('gap-0 overflow-hidden'):
                                        ui.label(f['filename']).classes('text-xs font-bold text-slate-200 truncate')
                                        ui.label(f'وضعیت: {f["status"]}').classes('text-[10px] text-emerald-400')

                                def delete_f(f_id=f['id']):
                                    stored_path = db.delete_file(f_id)
                                    if stored_path and os.path.exists(stored_path):
                                        try:
                                            os.remove(stored_path)
                                        except OSError:
                                            pass
                                    files_container.refresh()

                                ui.button(icon='delete', on_click=delete_f).props('flat round color=negative size=xs')

            files_container()

        # ۲. ستون چت (سمت راست - کاربر سمت راست، سیستم سمت چپ)
        with ui.card().classes('w-2/3 h-full bg-slate-900 border border-slate-800 p-0 flex flex-col justify-between rounded-2xl overflow-hidden'):

            # منطقه اسکرول پیام‌ها
            messages_scroll = ui.scroll_area().classes('w-full flex-grow p-4')

            def render_messages():
                messages_scroll.clear()
                with messages_scroll:
                    messages = db.get_messages(chat_id)
                    if not messages:
                        with ui.column().classes('w-full items-center justify-center py-20'):
                            ui.icon('chat_bubble_outline', size='48px').classes('text-slate-700 mb-2')
                            ui.label('سوال خود را درباره اسناد بالا بپرسید...').classes('text-slate-500 text-sm')

                    for msg in messages:
                        is_user = (msg['role'] == 'user')

                        # نکته: قبلاً فکر می‌کردیم justify-start توی محیط
                        # RTL یعنی سمت راست، ولی طبق چیزی که واقعاً روی
                        # صفحه دیدیم برعکس بود؛ پس مستقیم بر اساس نتیجه‌ی
                        # واقعی تنظیم شده: کاربر = justify-end (راست)،
                        # سیستم = justify-start (چپ).
                        align_cls = 'justify-end' if is_user else 'justify-start'

                        if is_user:
                            bg_cls = 'bg-sky-600 text-white rounded-br-none'
                            avatar = '👤'
                        else:
                            bg_cls = 'bg-slate-800 border border-slate-700 text-slate-100 rounded-bl-none'
                            avatar = '🤖'

                        def render_avatar():
                            avatar_cls = (
                                'bg-sky-700' if is_user
                                else 'bg-slate-800 border border-slate-700'
                            )
                            ui.label(avatar).classes(f'text-base p-1.5 {avatar_cls} rounded-full flex-shrink-0')

                        def render_bubble():
                            with ui.column().classes(f'max-w-[75%] {bg_cls} p-3.5 rounded-2xl shadow-md'):
                                ui.label(msg['content']).classes('text-sm leading-relaxed whitespace-pre-wrap')
                                ui.label(msg['created_at'].split()[1][:5]).classes('text-[10px] opacity-60 self-end mt-1')

                        with ui.row().classes(f'w-full {align_cls} my-2 items-start gap-2.5'):
                            # ترتیب فرزندها عمداً بر اساس is_user فرق
                            # می‌کنه تا آواتار همیشه توی لبه‌ی «بیرونی»
                            # باشه: برای کاربر (راست) آواتار بعد از حباب
                            # میاد (لبه‌ی راست)، برای سیستم (چپ) آواتار
                            # قبل از حباب میاد (لبه‌ی چپ).
                            if is_user:
                                render_bubble()
                                render_avatar()
                            else:
                                render_avatar()
                                render_bubble()

                    messages_scroll.scroll_to(percent=100)

            render_messages()

            # کادر ورودی پیام
            with ui.row().classes('w-full p-3 bg-slate-950/80 border-t border-slate-800 gap-2 items-center'):
                text_input = ui.input(placeholder='سوال خود را بنویسید... (Enter برای ارسال)').classes('flex-grow').props('dark rounded outlined dense')

                def send_msg():
                    val = text_input.value.strip()
                    if val:
                        db.add_message(chat_id, role='user', content=val)

                        # پاسخ هوش مصنوعی
                        files = db.get_files(chat_id)
                        if not files:
                            ans = "⚠️ لطفاً ابتدا حداقل یک فایل PDF در ستون سمت چپ بارگذاری کنید."
                        else:
                            ans = f"پاسخ هوش مصنوعی بر اساس {len(files)} سند آپلود شده برای سوال شما:\n«{val}»"

                        db.add_message(chat_id, role='assistant', content=ans)
                        text_input.value = ''
                        render_messages()

                text_input.on('keydown.enter', send_msg)
                ui.button(icon='send', on_click=send_msg).props('round color=primary').classes('shadow-lg shadow-sky-500/20')


# اجرای برنامه
ui.run(title='دستیار هوشمند اسناد PDF', port=8080, reload=True)
