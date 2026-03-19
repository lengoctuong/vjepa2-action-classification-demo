import gradio as gr

def test_connection():
    return "Kết nối thành công! Server Linux của bạn có thể bắn Public URL ra ngoài."

# Tạo một giao diện web cực kỳ đơn giản chỉ có 1 nút bấm
demo = gr.Interface(fn=test_connection, inputs=None, outputs="text")

# Tham số share=True là chìa khóa!
demo.launch(share=True)