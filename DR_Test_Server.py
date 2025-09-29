# DR_Server.py - 糖尿病视网膜病变诊断服务端
from DR_Test import graph  # 使用新的诊断图
import random
import gradio as gr
import time
import json
import os
from datetime import datetime

def process_dr_diagnosis(input_text, files):
    """
    处理糖尿病视网膜病变诊断请求
    """
    config = {
        "configurable": {
            "thread_id": f"dr_{random.randint(1000, 9999)}"
        }
    }
    
    try:
        # 构建诊断输入
        diagnosis_input = {
            "patient_query": input_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # 如果有文件，添加文件信息
        if files:
            file_info = []
            for file in files:
                file_name = os.path.basename(file.name) if hasattr(file, 'name') else os.path.basename(file)
                file_info.append(file_name)
            diagnosis_input["uploaded_files"] = file_info
        
        # 模拟分级模型结果（实际使用时替换为真实模型调用）
        # 这里根据输入文本模拟不同的分级结果
        if "轻度" in input_text or "1级" in input_text:
            model_grade = 1
        elif "中度" in input_text or "2级" in input_text:
            model_grade = 2
        elif "重度" in input_text or "3级" in input_text:
            model_grade = 3
        elif "增殖" in input_text or "4级" in input_text:
            model_grade = 4
        else:
            model_grade = random.randint(0, 2)  # 随机生成测试数据
        
        # 添加模拟的分级模型结果
        diagnosis_input.update({
            "model_grade": model_grade,
            "confidence": random.randint(75, 95),
            "image_path": "/data/retina_images/sample.jpg",
            "patient_info": {
                "age": 58,
                "diabetes_type": "2型", 
                "diabetes_duration": 10,
                "hbA1c": 7.5,
                "other_conditions": []
            }
        })
        
        # 调用诊断图
        result = graph.invoke(
            {"messages": [json.dumps(diagnosis_input, ensure_ascii=False)]}, 
            config
        )
        
        return result["messages"][-1].content
        
    except Exception as e:
        error_msg = f"诊断处理错误: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg

def generate_dr_report():
    """生成诊断报告文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dr_diagnosis_report_{timestamp}.json"
    
    sample_report = {
        "report_type": "糖尿病视网膜病变诊断报告",
        "generated_at": datetime.now().isoformat(),
        "note": "实际报告将包含完整的诊断数据"
    }
    
    # 保存临时文件（实际使用时保存真实报告）
    temp_path = f"/tmp/{filename}"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(sample_report, f, ensure_ascii=False, indent=2)
    
    return temp_path

def clear_files():
    return None

def clear_all():
    return "", "", None

# 使用之前提供的医疗主题CSS
medical_css = """
/* 这里放入之前提供的完整CSS代码 */
.gradio-container {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    min-height: 100vh;
    padding: 20px;
}
/* ... 其余CSS代码保持不变 ... */
"""

# 创建Gradio界面（使用之前提供的完整界面代码）
with gr.Blocks(css=medical_css, theme=gr.themes.Soft()) as demo:
    # 这里放入之前提供的完整Gradio界面代码
    # ...
    
    # 头部区域
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 🩺 糖尿病视网膜病变AI诊断系统")
        gr.Markdown("基于多模型集成的智能诊断平台")
    
    with gr.Row(equal_height=True, elem_classes="main-container"):
        # 左侧输入区域
        with gr.Column(scale=1, min_width=400, elem_classes="input-column"):
            gr.Markdown("### 📝 患者信息与诊断需求", elem_classes="section-title")
            inputs_text = gr.Textbox(
                label="",
                placeholder="请输入患者信息或诊断需求...",
                lines=4,
                max_lines=6,
                elem_classes="textbox"
            )

            # 快速开始示例
            with gr.Column(elem_classes="example-section"):
                gr.Markdown("### 🚀 快速诊断示例", elem_classes="example-title")
                with gr.Column(elem_classes="example-container"):
                    examples = gr.Examples(
                        examples=[
                            ["58岁女性，2型糖尿病8年，HbA1c 7.1%，请分析眼底病变程度"],
                            ["62岁男性，2型糖尿病15年，HbA1c 8.2%，高血压，请进行视网膜病变分级"],
                            ["45岁女性，1型糖尿病18年，近期视力模糊，请评估视网膜状况"]
                        ],
                        inputs=inputs_text,
                        examples_per_page=3
                    )

            # 按钮区域
            with gr.Row(elem_classes="buttons-row"):
                btn_clear = gr.Button("🗑️ 清除", variant="secondary", elem_classes="btn-secondary")
                btn_start = gr.Button("🔍 开始诊断", variant="primary", elem_classes="btn-primary")

        # 右侧输出区域
        with gr.Column(scale=1, min_width=500, elem_classes="output-column"):
            gr.Markdown("### 📊 诊断结果报告", elem_classes="section-title")
            outputs_text = gr.Textbox(
                label="",
                interactive=False,
                lines=20,
                show_copy_button=True,
                elem_classes="output-textbox"
            )
            
            with gr.Row(elem_classes="buttons-row"):
                btn_download = gr.Button("💾 导出报告", variant="secondary", elem_classes="btn-secondary")
                btn_new = gr.Button("🆕 新建诊断", variant="secondary", elem_classes="btn-secondary")

    # 事件处理
    btn_start.click(
        fn=process_dr_diagnosis,
        inputs=[inputs_text, gr.File(visible=False)],  # 暂时禁用文件上传
        outputs=[outputs_text]
    )
    
    btn_download.click(
        fn=generate_dr_report,
        inputs=[],
        outputs=[gr.File(label="下载诊断报告")]
    )
    
    def clear_input():
        return ""
    
    btn_clear.click(
        fn=clear_input,
        inputs=[],
        outputs=[inputs_text]
    )
    
    btn_new.click(
        fn=clear_all,
        inputs=[],
        outputs=[inputs_text, outputs_text]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )