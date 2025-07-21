import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
import numpy as np
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Загрузка модели
try:
    model = tf.keras.models.load_model(r'models\rorovaa_model_flowers.keras')
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    raise

# Параметры классификации
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
class_colors = ['#FFD700', '#FFFFFF', '#FF00FF', '#FFA500', '#FF007F']

# Создаем главное окно
root = tk.Tk()
root.title("🌸 FlowerVision PRO - AI Flower Classifier")
root.geometry("1200x800")
root.configure(bg='#0A0E17')
root.resizable(True, True)

# Переменная для полноэкранного режима
fullscreen = False

# Функция переключения полноэкранного режима
def toggle_fullscreen(event=None):
    global fullscreen
    fullscreen = not fullscreen
    root.attributes('-fullscreen', fullscreen)
    if not fullscreen:
        root.geometry("1200x800")

# Выход по Esc
root.bind('<Escape>', toggle_fullscreen)

# Стили для виджетов
style = ttk.Style()
style.theme_use('clam')

# Настраиваем стили
style.configure('TFrame', background='#0A0E17')
style.configure('TLabel', background='#0A0E17', foreground='#ECF0F1', font=('Helvetica', 10))
style.configure('Header.TLabel', font=('Montserrat', 16, 'bold'), foreground='#7B68EE')
style.configure('Result.TLabel', font=('Montserrat', 14, 'bold'), foreground='#FF69B4')
style.configure('TButton', font=('Montserrat', 12), borderwidth=0, focuscolor='none')
style.map('TButton', 
          background=[('active', '#5D3FD3'), ('pressed', '#4B0082')],
          foreground=[('active', 'white')])

# Градиентный фон
bg_canvas = tk.Canvas(root, bg='#0A0E17', highlightthickness=0)
bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

# Главный контейнер
main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Заголовок
header_frame = ttk.Frame(main_frame)
header_frame.pack(fill=tk.X, pady=(0, 20))

header_label = ttk.Label(header_frame, 
                         text="🌺 FLOWERVISION PRO - AI POWERED FLOWER CLASSIFIER", 
                         style='Header.TLabel')
header_label.pack(side=tk.LEFT)

# Кнопка полноэкранного режима
fullscreen_btn = ttk.Button(header_frame, text="⛶", 
                           command=toggle_fullscreen, 
                           width=3)
fullscreen_btn.pack(side=tk.RIGHT, padx=5)

# Основное содержимое
content_frame = ttk.Frame(main_frame)
content_frame.pack(fill=tk.BOTH, expand=True)

# Левая панель (изображение)
left_frame = ttk.Frame(content_frame, width=500)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

# Холст для изображения с эффектом стекла
img_frame = ttk.Frame(left_frame)
img_frame.pack(fill=tk.BOTH, expand=True)

img_canvas = tk.Canvas(img_frame, bg='#1A1F2C', highlightthickness=0, bd=0)
img_canvas.pack(fill=tk.BOTH, expand=True)

# Стеклянный эффект
img_canvas.create_rectangle(10, 10, 490, 490, fill='#1A1F2C', outline='', stipple='gray25')
img_canvas.create_rectangle(15, 15, 485, 485, fill='#0A0E17', outline='#7B68EE', width=2)

img_label = ttk.Label(img_canvas)
img_label.place(relx=0.5, rely=0.5, anchor='center')

# Кнопка загрузки изображения (большая и стильная)
upload_btn_frame = ttk.Frame(left_frame)
upload_btn_frame.pack(fill=tk.X, pady=10)

def create_upload_button():
    btn = tk.Canvas(upload_btn_frame, width=300, height=60, bg='#0A0E17', highlightthickness=0)
    
    # Рисуем неоновую кнопку
    btn.create_rectangle(5, 5, 295, 55, 
                        fill='#0A0E17', 
                        outline='#7B68EE', 
                        width=3)
    btn.create_text(150, 30, 
                   text="📁 UPLOAD FLOWER IMAGE", 
                   fill='#7B68EE', 
                   font=('Montserrat', 12, 'bold'))
    
    # Эффекты при наведении
    def on_enter(e):
        btn.delete('all')
        btn.create_rectangle(5, 5, 295, 55, 
                            fill='#7B68EE', 
                            outline='#7B68EE', 
                            width=3)
        btn.create_text(150, 30, 
                       text="📁 UPLOAD FLOWER IMAGE", 
                       fill='white', 
                       font=('Montserrat', 12, 'bold'))
    
    def on_leave(e):
        btn.delete('all')
        btn.create_rectangle(5, 5, 295, 55, 
                            fill='#0A0E17', 
                            outline='#7B68EE', 
                            width=3)
        btn.create_text(150, 30, 
                       text="📁 UPLOAD FLOWER IMAGE", 
                       fill='#7B68EE', 
                       font=('Montserrat', 12, 'bold'))
    
    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)
    btn.bind('<Button-1>', lambda e: open_file())
    return btn

upload_btn = create_upload_button()
upload_btn.pack(pady=10)

# Правая панель (результаты)
right_frame = ttk.Frame(content_frame, width=500)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Результаты классификации
result_frame = ttk.LabelFrame(right_frame, text="AI ANALYSIS RESULTS", padding=15)
result_frame.pack(fill=tk.X, pady=(0, 20))

result_var = tk.StringVar(value="Upload an image to begin analysis")
result_label = ttk.Label(result_frame, textvariable=result_var, style='Result.TLabel')
result_label.pack(pady=15)

# Прогресс-бар
confidence_var = tk.DoubleVar()
confidence_bar = ttk.Progressbar(result_frame, variable=confidence_var, maximum=1.0, length=300)
confidence_bar.pack(pady=10)

# Гистограмма
fig = plt.Figure(figsize=(5, 3), facecolor='#0A0E17')
ax = fig.add_subplot(111)
ax.set_facecolor('#0A0E17')
ax.tick_params(colors='#7B68EE')
canvas = FigureCanvasTkAgg(fig, master=result_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Таблица результатов
tree = ttk.Treeview(right_frame, columns=('probability'), height=5)
tree.heading('#0', text='FLOWER CLASS')
tree.heading('probability', text='PROBABILITY')
tree.column('#0', width=200)
tree.column('probability', width=150)
tree.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

# Функция загрузки изображения
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        load_image(file_path)

def load_image(path):
    try:
        pil_img = Image.open(path)
        
        # Создаем эффект стеклянной карточки
        w, h = pil_img.size
        max_size = 450
        ratio = min(max_size/w, max_size/h)
        new_size = (int(w*ratio), int(h*ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        
        # Добавляем тень и скругленные углы
        mask = Image.new('L', new_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), new_size], 20, fill=255)
        
        shadow = Image.new('RGBA', (new_size[0]+10, new_size[1]+10), (0,0,0,0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle([(5,5), (new_size[0]+5, new_size[1]+5)], 20, fill=(123,104,238,150))
        shadow = shadow.filter(ImageFilter.GaussianBlur(10))
        
        final_img = Image.new('RGBA', (new_size[0]+10, new_size[1]+10), (0,0,0,0))
        final_img.paste(shadow, (0,0), shadow)
        final_img.paste(pil_img, (5,5), mask)
        
        tk_img = ImageTk.PhotoImage(final_img)
        img_label.configure(image=tk_img)
        img_label.image = tk_img
        
        # Запускаем предсказание
        threading.Thread(target=predict_flower, args=(path,), daemon=True).start()
        
    except Exception as e:
        result_var.set(f"Error: {str(e)}")

def predict_flower(path):
    try:
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [180, 180])
        img = img / 255.0
        img_array = tf.expand_dims(img, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        update_ui(predictions)
        
    except Exception as e:
        root.after(0, lambda: result_var.set(f"Prediction error: {str(e)}"))

def update_ui(predictions):
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Анимация прогресс-бара
    for i in range(101):
        confidence_var.set(i/100 * confidence)
        root.update()
        time.sleep(0.01)
    
    result_var.set(f"PREDICTION: {class_names[predicted_idx].upper()} ({confidence:.2%})")
    
    # Обновление гистограммы
    ax.clear()
    bars = ax.bar(class_names, predictions, color=class_colors)
    ax.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='white')
    
    canvas.draw()
    
    # Обновление таблицы
    tree.delete(*tree.get_children())
    for i, prob in enumerate(predictions):
        tree.insert("", "end", text=class_names[i].upper(), values=(f"{prob:.2%}"))
    
    tree.tag_configure('predicted', background='#1A1F2C', foreground='#FF69B4')
    tree.item(tree.get_children()[predicted_idx], tags=('predicted',))

# Кнопка выхода
exit_btn_frame = ttk.Frame(main_frame)
exit_btn_frame.pack(fill=tk.X, pady=(10, 0))

def create_exit_button():
    btn = tk.Canvas(exit_btn_frame, width=150, height=50, bg='#0A0E17', highlightthickness=0)
    
    btn.create_rectangle(5, 5, 145, 45, 
                        fill='#0A0E17', 
                        outline='#FF5555', 
                        width=2)
    btn.create_text(75, 25, 
                   text="🚪 EXIT", 
                   fill='#FF5555', 
                   font=('Montserrat', 10, 'bold'))
    
    def on_enter(e):
        btn.delete('all')
        btn.create_rectangle(5, 5, 145, 45, 
                            fill='#FF5555', 
                            outline='#FF5555', 
                            width=2)
        btn.create_text(75, 25, 
                       text="🚪 EXIT", 
                       fill='white', 
                       font=('Montserrat', 10, 'bold'))
    
    def on_leave(e):
        btn.delete('all')
        btn.create_rectangle(5, 5, 145, 45, 
                            fill='#0A0E17', 
                            outline='#FF5555', 
                            width=2)
        btn.create_text(75, 25, 
                       text="🚪 EXIT", 
                       fill='#FF5555', 
                       font=('Montserrat', 10, 'bold'))
    
    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)
    btn.bind('<Button-1>', lambda e: root.destroy())
    return btn

exit_btn = create_exit_button()
exit_btn.pack(side=tk.RIGHT, padx=10)

root.mainloop()
