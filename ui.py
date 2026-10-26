import os
import time
import asyncio
from nicegui import ui
import chat_database as db
from orchestrator import get_orchestrator

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
# راهنمای استفاده (دیالوگ مشترک)
# ============================================================
def open_help_dialog():
    """راهنمای استفاده از برنامه."""
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
                    'گفتگو انتخاب کن. هر گفتگو کاملاً مستقل از بقیه‌ست.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۲. بارگذاری اسناد PDF', icon='upload_file').classes('w-full text-slate-200'):
                ui.label(
                    'داخل هر گفتگو، از ستون «اسناد این گفتگو» فایل‌های PDF مورد نظرت '
                    'رو آپلود کن.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۳. پرسیدن سوال', icon='chat').classes('w-full text-slate-200'):
                ui.label(
                    'بعد از بارگذاری حداقل یک فایل، سوالت رو توی کادر پایین بنویس.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۴. مشاهده‌ی مرجع پاسخ', icon='find_in_page').classes('w-full text-slate-200'):
                ui.label(
                    'کنار هر پاسخ، شماره‌ی صفحه‌ای که اطلاعات ازش استخراج شده نشون داده می‌شه.'
                ).classes('text-sm text-slate-400 leading-relaxed')

            with ui.expansion('۵. تغییر نام / حذف گفتگو', icon='edit').classes('w-full text-slate-200'):
                ui.label(
                    'از دکمه‌ی مداد بالای هر گفتگو می‌تونی اسمش رو عوض کنی.'
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

                async def save():
                    if title_input.value:
                        new_id = db.create_chat(title=title_input.value.strip(), color=selected_color['hex'])
                        # آماده‌سازی پوشه دیسک و کالکشن ChromaDB در ترد پس‌زمینه بدون قفل کردن رابط کاربری
                        db.get_chat_data_dir(new_id)
                        orch = get_orchestrator()
                        await asyncio.to_thread(orch.vector_db.get_or_create_collection, new_id)
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

        search_input = ui.input(placeholder='🔍 جستجو در عنوان گفتگوها...').classes('w-full max-w-md mx-auto').props('dark rounded outlined dense [dir=rtl]')

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
                            ui.element('div').classes('h-1.5 w-full rounded-full mb-4').style(f'background-color: {chat["color"]}; box-shadow: 0 0 10px {chat["color"]}aa;')

                            ui.label(chat['title']).classes('text-xl font-bold text-slate-100 line-clamp-1 mb-2')

                            with ui.row().classes('items-center gap-2 text-slate-400 text-xs mb-1'):
                                ui.icon('schedule', size='16px')
                                ui.label(f'ایجاد: {chat["created_at"]}')

                            with ui.row().classes('w-full justify-between items-center mt-6 pt-3 border-t border-slate-800/60'):
                                def confirm_delete(chat_to_del=chat):
                                    with ui.dialog() as d, ui.card().classes('bg-slate-900 border border-slate-800 p-5'):
                                        ui.label(f'آیا از حذف گفتگوی «{chat_to_del["title"]}» مطمئن هستید؟').classes('text-slate-200 font-bold')
                                        with ui.row().classes('w-full justify-end gap-2 mt-4'):
                                            ui.button('انصراف', on_click=d.close).props('flat color=grey')

                                            async def do_delete():
                                                orch = get_orchestrator()
                                                await asyncio.to_thread(orch.delete_chat, chat_to_del['id'])
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

            upload_status = {'notification': None}

            def handle_begin_upload():
                upload_status['notification'] = ui.notification(
                    message='📤 در حال آپلود فایل...',
                    spinner=True,
                    type='ongoing',
                    timeout=None,
                )

            def finish_upload_notification(message: str, success: bool) -> None:
                notification = upload_status['notification']
                if notification is None:
                    ui.notify(message, type='positive' if success else 'negative')
                    return
                notification.spinner = False
                notification.message = message
                notification.type = 'positive' if success else 'negative'
                notification.timeout = 3.0

            async def read_file_bytes(file_obj):
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

            async def handle_upload(e):
                try:
                    file_name = e.file.name

                    if not file_name.lower().endswith('.pdf'):
                        finish_upload_notification('❌ فقط فایل‌های با فرمت PDF مجاز هستند!', success=False)
                        return

                    existing_names = {f['filename'] for f in db.get_files(chat_id)}
                    if file_name in existing_names:
                        finish_upload_notification(f'⚠️ «{file_name}» قبلاً به این گفتگو اضافه شده.', success=False)
                        return

                    chat_dir = db.get_chat_data_dir(chat_id)
                    save_path = chat_dir / file_name

                    content_bytes = await read_file_bytes(e.file)

                    with open(save_path, 'wb') as f:
                        f.write(content_bytes)

                    file_record_id = db.add_file(chat_id=chat_id, filename=file_name, stored_path=str(save_path), status='processing')
                    files_container.refresh()

                    # اجرای پایپ‌لاین پیش‌پردازش و امبدینگ در ارکستراتور
                    orch = get_orchestrator()

                    def progress_callback(msg: str, pct: float):
                        notif = upload_status.get('notification')
                        if notif:
                            notif.message = f"⏳ {msg} ({int(pct * 100)}%)"

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        orch.process_document,
                        save_path,
                        chat_id,
                        file_name,
                        progress_callback
                    )

                    db.update_file_status(file_record_id, 'done')
                    finish_upload_notification(f'✅ پردازش سند «{file_name}» با موفقیت تکمیل شد', success=True)
                    files_container.refresh()

                except Exception as ex:
                    finish_upload_notification(f'❌ خطا در پردازش سند: {ex}', success=False)

                finally:
                    uploader.reset()
                    upload_status['notification'] = None

            uploader = ui.upload(
                on_begin_upload=handle_begin_upload,
                on_upload=handle_upload,
                label='انتخاب فایل PDF',
                auto_upload=True
            ).classes('w-full mb-3').props('dark flat bordered accept=".pdf"')

            ui.separator().classes('bg-slate-800 my-2')

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

                                async def delete_f(f_id=f['id'], f_name=f['filename']):
                                    orch = get_orchestrator()
                                    await asyncio.to_thread(orch.delete_document, chat_id, f_name, f_id)
                                    files_container.refresh()

                                ui.button(icon='delete', on_click=delete_f).props('flat round color=negative size=xs')

            files_container()

        # ۲. ستون چت (سمت راست)
        with ui.card().classes('w-2/3 h-full bg-slate-900 border border-slate-800 p-0 flex flex-col justify-between rounded-2xl overflow-hidden'):

            messages_scroll = ui.scroll_area().classes('w-full flex-grow p-4')
            with messages_scroll:
                messages_container = ui.column().classes('w-full gap-2')

            is_user_at_bottom = True

            def handle_scroll(e):
                nonlocal is_user_at_bottom
                # اگر کاربر در انتهای صفحه (بیش از ۸۸٪) باشد دنبال کردن زنده فعال است
                # اگر کاربر خودش به بالا اسکرول کند، به خواندن پیام‌های قبلی احترام گذاشته می‌شود
                if hasattr(e, 'vertical_percentage'):
                    is_user_at_bottom = (e.vertical_percentage >= 0.88)

            messages_scroll.on_scroll(handle_scroll)

            empty_placeholder = None

            reply_state = {'text': None}

            reply_bar = ui.row().classes('w-full px-4 py-1.5 bg-slate-900/90 border-t border-sky-500/30 items-center justify-between text-xs text-sky-400')
            reply_bar.set_visibility(False)
            with reply_bar:
                with ui.row().classes('items-center gap-2 overflow-hidden flex-grow'):
                    ui.icon('reply', size='16px').classes('text-sky-400')
                    reply_preview_label = ui.label('').classes('truncate text-slate-300 font-medium')
                def cancel_reply():
                    reply_state['text'] = None
                    reply_bar.set_visibility(False)
                ui.button(icon='close', on_click=cancel_reply).props('flat round size=xs color=grey').classes('hover:text-red-400')

            def set_reply(target_text: str):
                cleaned = target_text.strip()
                if not cleaned:
                    return
                reply_state['text'] = cleaned
                preview = cleaned[:75] + ('...' if len(cleaned) > 75 else '')
                reply_preview_label.text = f'در پاسخ به: {preview}'
                reply_bar.set_visibility(True)
                try:
                    text_input.run_method('focus')
                except Exception:
                    pass

            def append_bubble(role: str, content: str, time_str: str = None, reply_to_text: str = None):
                nonlocal empty_placeholder
                if empty_placeholder is not None:
                    try:
                        empty_placeholder.delete()
                    except Exception:
                        pass
                    empty_placeholder = None

                is_user = (role == 'user')
                align_cls = 'justify-end' if is_user else 'justify-start'
                bg_cls = 'bg-sky-600 text-white rounded-br-none' if is_user else 'bg-slate-800 border border-slate-700 text-slate-100 rounded-bl-none'
                avatar = '👤' if is_user else '🤖'
                avatar_bg = 'bg-sky-700' if is_user else 'bg-slate-800 border border-slate-700'
                t_str = time_str or (db.get_current_local_time_str().split()[1][:5] if hasattr(db, 'get_current_local_time_str') else '')

                with messages_container:
                    with ui.row().classes(f'w-full {align_cls} my-1 items-start gap-2.5 group'):
                        if is_user:
                            with ui.column().classes(f'max-w-[75%] {bg_cls} p-3.5 rounded-2xl shadow-md'):
                                if reply_to_text:
                                    rep_snip = reply_to_text[:65] + ('...' if len(reply_to_text) > 65 else '')
                                    with ui.row().classes('items-center gap-1.5 border-r-2 border-sky-300 pr-2 mb-2 bg-sky-700/50 p-1.5 rounded-lg text-xs text-sky-100 w-full overflow-hidden'):
                                        ui.icon('reply', size='12px').classes('opacity-80')
                                        ui.label(rep_snip).classes('truncate')
                                ui.label(f"\u200f{content}").classes('text-sm leading-relaxed whitespace-pre-wrap text-right').style('direction: rtl; unicode-bidi: plaintext;')
                                if t_str:
                                    ui.label(t_str).classes('text-[10px] opacity-60 self-end mt-1')
                            ui.label(avatar).classes(f'text-base p-1.5 {avatar_bg} rounded-full flex-shrink-0')
                        else:
                            ui.label(avatar).classes(f'text-base p-1.5 {avatar_bg} rounded-full flex-shrink-0')
                            with ui.column().classes(f'max-w-[75%] {bg_cls} p-3.5 rounded-2xl shadow-md'):
                                ui.label(f"\u200f{content}").classes('text-sm leading-relaxed whitespace-pre-wrap text-right').style('direction: rtl; unicode-bidi: plaintext;')
                                with ui.row().classes('w-full items-center justify-between mt-1 pt-1 border-t border-slate-700/40'):
                                    ui.button(icon='reply', on_click=lambda t=content: set_reply(t)).props('flat round size=xs color=grey').classes('opacity-50 hover:opacity-100 transition-opacity').tooltip('پاسخ (ریپلای) به این پیام')
                                    if t_str:
                                        ui.label(t_str).classes('text-[10px] opacity-60')

            def init_messages():
                messages = db.get_messages(chat_id)
                if not messages:
                    nonlocal empty_placeholder
                    with messages_container:
                        with ui.column().classes('w-full items-center justify-center py-20') as placeholder:
                            empty_placeholder = placeholder
                            ui.icon('chat_bubble_outline', size='48px').classes('text-slate-700 mb-2')
                            ui.label('سوال خود را درباره اسناد بالا بپرسید...').classes('text-slate-500 text-sm')
                else:
                    for msg in messages:
                        t = msg['created_at'].split()[1][:5] if msg.get('created_at') else ''
                        append_bubble(msg['role'], msg['content'], t)
                    messages_scroll.scroll_to(percent=100)

            init_messages()

            # کادر ورودی پیام
            with ui.row().classes('w-full p-3 bg-slate-950/80 border-t border-slate-800 gap-2 items-center'):
                text_input = ui.textarea(placeholder='سوال خود را بنویسید... (Enter برای ارسال، Shift+Enter برای خط بعد)').classes('flex-grow').props('dark rounded outlined dense autogrow rows=1 [dir=rtl]').style('direction: rtl; unicode-bidi: plaintext;')
                send_btn = ui.button(icon='send').props('round color=primary').classes('shadow-lg shadow-sky-500/20')

                async def send_msg():
                    nonlocal is_user_at_bottom
                    val = text_input.value.strip()
                    if not val:
                        return

                    # استخراج زمینه ریپلای و پاکسازی نوار ریپلای
                    current_reply_text = reply_state['text']
                    cancel_reply()

                    text_input.value = ''
                    send_btn.disable()
                    text_input.disable()

                    # با ارسال پیام جدید، اسکرول را پایین می‌بریم و ردیابی را فعال می‌کنیم
                    is_user_at_bottom = True

                    # ثبت پیام در دیتابیس و اضافه کردن حباب کاربر با نماد ریپلای
                    db.add_message(chat_id, role='user', content=val)
                    append_bubble('user', val, reply_to_text=current_reply_text)
                    messages_scroll.scroll_to(percent=100)

                    files = db.get_files(chat_id)
                    if not files:
                        no_file_ans = "لطفا حداقل یک سند بارگذاری کنید تا بر اساس آن به سوالاتتان پاسخ دهیم ⚠️"
                        db.add_message(chat_id, role='assistant', content=no_file_ans)
                        append_bubble('assistant', no_file_ans)
                        messages_scroll.scroll_to(percent=100)
                        send_btn.enable()
                        text_input.enable()
                        return

                    # بررسی گارد پردازش اسناد (In-Flight Processing Guard)
                    is_processing = any(f.get('status') == 'processing' for f in files)
                    if is_processing:
                        guard_ans = "⚠️ در حال پردازش اسناد .... لطفاً منتظر بمانید"
                        db.add_message(chat_id, role='assistant', content=guard_ans)
                        append_bubble('assistant', guard_ans)
                        messages_scroll.scroll_to(percent=100)
                        send_btn.enable()
                        text_input.enable()
                        return

                    # ساخت حباب پیام استریم با کاراکتر RLM و تنظیمات BiDi
                    with messages_container:
                        with ui.row().classes('w-full justify-start my-1 items-start gap-2.5 group'):
                            ui.label('🤖').classes('text-base p-1.5 bg-slate-800 border border-slate-700 rounded-full flex-shrink-0')
                            with ui.column().classes('max-w-[75%] bg-slate-800 border border-slate-700 text-slate-100 rounded-bl-none p-3.5 rounded-2xl shadow-md'):
                                response_label = ui.label('\u200f').classes('text-sm leading-relaxed whitespace-pre-wrap text-right').style('direction: rtl; unicode-bidi: plaintext;')
                                footer_row = ui.row().classes('w-full items-center justify-between mt-1 pt-1 border-t border-slate-700/40')
                                with footer_row:
                                    reply_action_btn = ui.button(icon='reply').props('flat round size=xs color=grey').classes('opacity-50 hover:opacity-100 transition-opacity').tooltip('پاسخ (ریپلای) به این پیام')
                                    time_label = ui.label('').classes('text-[10px] opacity-60')

                    messages_scroll.scroll_to(percent=100)

                    # قفل ایمن و دریافت استریم پاسخ از ارکستراتور با زمینه ریپلای
                    try:
                        text_input.props('loading')
                        text_input.placeholder = "در حال تولید پاسخ هوشمند..."

                        orch = get_orchestrator()
                        full_response = ""
                        citations_list = []
                        last_scroll_time = 0.0

                        async for packet in orch.ask_question_stream(val, chat_id, reply_context=current_reply_text):
                            p_type = packet.get("type")
                            if p_type == "token":
                                tok = packet.get("content", "")
                                full_response += tok
                                response_label.text = f"\u200f{full_response}"

                                # کنترل هوشمند اسکرول: فقط در صورتی که کاربر خودش به بالا اسکرول نکرده باشد
                                now = time.time()
                                if is_user_at_bottom and (now - last_scroll_time > 0.3):
                                    messages_scroll.scroll_to(percent=100)
                                    last_scroll_time = now

                            elif p_type == "sources":
                                citations_list = packet.get("citations", [])

                        # پیوست مراجع استنادی در صورت وجود
                        if citations_list:
                            sources_text = "\n\n📌 **منابع و مراجع سند:**\n"
                            seen = set()
                            for c in citations_list:
                                fname = c.get("filename", "")
                                pnum = c.get("page_number", "")
                                ref_key = f"{fname}_{pnum}"
                                if ref_key not in seen:
                                    seen.add(ref_key)
                                    sources_text += f"- 📄 سند: `{fname}` (صفحه {pnum})\n"
                            full_response += sources_text
                            response_label.text = f"\u200f{full_response}"

                        # فعال‌سازی ریپلای برای پاسخ تازه تولیدشده
                        captured_res = full_response
                        reply_action_btn.on_click(lambda: set_reply(captured_res))
                        time_label.text = db.get_current_local_time_str().split()[1][:5] if hasattr(db, 'get_current_local_time_str') else ''

                        if is_user_at_bottom:
                            messages_scroll.scroll_to(percent=100)

                        # ذخیره پاسخ کامل در دیتابیس پیام‌های چت SQLite
                        db.add_message(chat_id, role='assistant', content=full_response)

                    except Exception as ex:
                        err_msg = f"❌ متأسفانه در فرآیند تولید پاسخ خطایی رخ داد: {ex}"
                        db.add_message(chat_id, role='assistant', content=err_msg)
                        append_bubble('assistant', err_msg)
                        messages_scroll.scroll_to(percent=100)

                    finally:
                        text_input.props(remove='loading')
                        text_input.placeholder = "سوال خود را بنویسید... (Enter برای ارسال، Shift+Enter برای خط بعد)"
                        send_btn.enable()
                        text_input.enable()
                        try:
                            text_input.run_method('focus')
                        except Exception:
                            pass

                text_input.on('keydown.enter.exact.prevent', send_msg)
                text_input.on('keydown.enter.shift', lambda e: None)
                send_btn.on('click', send_msg)


import config


def start_ui_server(port: int = None, reload: bool = False):
    """تابع راه‌اندازی سرور رابط کاربری NiceGUI."""
    srv_port = port if port is not None else getattr(config, 'SERVER_PORT', 8050)
    ui.run(title='دستیار هوشمند اسناد PDF', port=srv_port, reload=reload)


if __name__ in {"__main__", "__mp_main__"}:
    start_ui_server()
