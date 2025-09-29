# DR_Server.py - 简约版糖尿病视网膜病变诊断服务端
from DR_Test import graph
import random
import gradio as gr
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
        
        # 模拟分级模型结果
        if "轻度" in input_text or "1级" in input_text:
            model_grade = 1
        elif "中度" in input_text or "2级" in input_text:
            model_grade = 2
        elif "重度" in input_text or "3级" in input_text:
            model_grade = 3
        elif "增殖" in input_text or "4级" in input_text:
            model_grade = 4
        else:
            model_grade = random.randint(0, 2)
        
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
        return f"诊断处理错误: {str(e)}"

def generate_dr_report():
    """生成诊断报告文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dr_diagnosis_report_{timestamp}.json"
    
    sample_report = {
        "report_type": "糖尿病视网膜病变诊断报告",
        "generated_at": datetime.now().isoformat(),
        "note": "实际报告将包含完整的诊断数据"
    }
    
    temp_path = f"/tmp/{filename}"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(sample_report, f, ensure_ascii=False, indent=2)
    
    return temp_path

def clear_all():
    return "", "", None

# 简约CSS样式
simple_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: #f8fafc;
    padding: 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
    padding: 30px 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
}

.main-title {
    color: #1e40af;
    font-size: 2.2em;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    color: #64748b;
    font-size: 1.1em;
}

.main-container {
    gap: 20px;
}

.input-column, .output-column {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
}

.section-title {
    color: #1e40af;
    font-weight: 600;
    margin-bottom: 15px;
    font-size: 1.2em;
}

.textbox {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 12px;
    font-size: 14px;
    background: #f9fafb;
}

.textbox:focus {
    border-color: #3b82f6;
    background: white;
}

.btn-primary {
    background: #3b82f6;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    color: white;
    font-weight: 500;
}

.btn-primary:hover {
    background: #2563eb;
}

.btn-secondary {
    background: white;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 10px 20px;
    color: #374151;
}

.btn-secondary:hover {
    border-color: #3b82f6;
    color: #1e40af;
}

.output-textbox {
    min-height: 400px;
    background: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    padding: 16px;
    font-size: 14px;
    line-height: 1.6;
}

.medical-alert {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 12px;
    margin: 15px 0;
    color: #92400e;
    font-size: 0.9em;
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: #6b7280;
    font-size: 0.9em;
    padding: 15px;
    background: white;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
"""

# 创建简约界面
with gr.Blocks(css=simple_css, theme=gr.themes.Soft()) as demo:
    
    # 头部
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 糖尿病视网膜病变AI诊断系统")
        gr.Markdown("基于多模型集成的智能诊断平台")
    
    with gr.Row(elem_classes="main-container"):
        # 左侧输入
        with gr.Column(scale=1, min_width=400, elem_classes="input-column"):
            gr.Markdown("### 患者信息", elem_classes="section-title")
            inputs_text = gr.Textbox(
                label="",
                placeholder="请输入患者信息...\n示例：62岁男性，2型糖尿病15年，HbA1c 8.2%",
                lines=4,
                elem_classes="textbox"
            )

            # 警示信息
            with gr.Column(elem_classes="medical-alert"):
                gr.Markdown("**提示**: 本系统为AI辅助诊断工具，结果仅供参考。")
            
            # 示例
            with gr.Column():
                gr.Markdown("**快速示例**")
                examples = gr.Examples(
                    examples=[
                        ["58岁女性，2型糖尿病8年，HbA1c 7.1%"],
                        ["62岁男性，2型糖尿病15年，HbA1c 8.2%，高血压"],
                        ["45岁女性，1型糖尿病18年，近期视力模糊"]
                    ],
                    inputs=inputs_text,
                    label=""
                )

            # 按钮
            with gr.Row():
                btn_start = gr.Button("开始诊断", variant="primary", elem_classes="btn-primary")
                btn_clear = gr.Button("清除", variant="secondary", elem_classes="btn-secondary")
        
        # 右侧输出
        with gr.Column(scale=1, min_width=500, elem_classes="output-column"):
            gr.Markdown("### 诊断报告", elem_classes="section-title")
            outputs_text = gr.Textbox(
                label="",
                interactive=False,
                lines=18,
                show_copy_button=True,
                elem_classes="output-textbox"
            )
            
            with gr.Row():
                btn_download = gr.Button("导出报告", variant="secondary", elem_classes="btn-secondary")
                btn_new = gr.Button("新建诊断", variant="secondary", elem_classes="btn-secondary")

    # 底部
    with gr.Column(elem_classes="footer"):
        gr.Markdown("糖尿病视网膜病变AI诊断系统 · 多模型集成 · 智能分级")
    
    # 事件处理
    btn_start.click(
        fn=process_dr_diagnosis,
        inputs=[inputs_text, gr.File(visible=False)],
        outputs=[outputs_text]
    )
    
    btn_download.click(
        fn=generate_dr_report,
        inputs=[],
        outputs=[gr.File(label="下载报告")]
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
        outputs=[inputs_text, outputs_text, gr.File(visible=False)]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
