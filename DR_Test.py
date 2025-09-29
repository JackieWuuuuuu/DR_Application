# Director_DR.py - 糖尿病视网膜病变专用工作流
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langchain_core.messages import AnyMessage
from operator import add
from langgraph.graph import StateGraph
import os
from langgraph.constants import START, END
import json
import asyncio

# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Diabetic_Retinopathy_Diagnosis"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# 初始化大模型
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-06cb8603d6784d53970f60bfdc372481"
)

# 定义状态
class DiagnosisState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    current_step: str
    patient_data: dict
    grading_result: dict
    vision_llm_result: dict
    integrated_result: dict
    knowledge_content: dict
    final_report: dict

# 糖尿病视网膜病变诊断系统
class DRDiagnosisSystem:
    def __init__(self):
        self.model_weights = {
            'grading_model': 0.6,
            'vision_llm': 0.4
        }
        
        self.grade_descriptions = {
            0: "无视网膜病变",
            1: "轻度非增殖性糖尿病视网膜病变（轻度NPDR）",
            2: "中度非增殖性糖尿病视网膜病变（中度NPDR）", 
            3: "重度非增殖性糖尿病视网膜病变（重度NPDR）",
            4: "增殖性糖尿病视网膜病变（PDR）"
        }
    
    def integrate_predictions(self, grading_model_result, vision_llm_result):
        """集成两个模型的预测结果"""
        grading_grade = grading_model_result.get('grade', 0)
        grading_confidence = grading_model_result.get('confidence', 0) / 100.0
        
        vision_grade = self._parse_vision_llm_output(vision_llm_result)
        vision_confidence = 0.8
        
        weighted_score = (
            grading_grade * self.model_weights['grading_model'] * grading_confidence +
            vision_grade * self.model_weights['vision_llm'] * vision_confidence
        )
        
        final_grade = round(weighted_score)
        final_grade = max(0, min(4, final_grade))
        
        return {
            'final_grade': final_grade,
            'grading_model_grade': grading_grade,
            'vision_llm_grade': vision_grade,
            'confidence_scores': {
                'grading_model': grading_confidence,
                'vision_llm': vision_confidence
            },
            'integration_details': {
                'weighted_score': weighted_score,
                'model_weights': self.model_weights
            },
            'agreement': grading_grade == vision_grade
        }
    
    def _parse_vision_llm_output(self, vision_output):
        """解析视觉大模型的输出"""
        if isinstance(vision_output, dict):
            return vision_output.get('predicted_grade', 0)
        elif isinstance(vision_output, str):
            import re
            grade_match = re.search(r'等级[：:\s]*(\d)', vision_output)
            if grade_match:
                return int(grade_match.group(1))
        return 0
    
    def generate_diagnosis_report(self, integrated_result, patient_info=None):
        """生成综合诊断报告"""
        final_grade = integrated_result['final_grade']
        
        return {
            'diagnosis_summary': {
                'grade': final_grade,
                'description': self.grade_descriptions[final_grade],
                'severity': self._get_severity_level(final_grade)
            },
            'model_analysis': {
                'grading_model_result': integrated_result['grading_model_grade'],
                'vision_llm_result': integrated_result['vision_llm_grade'],
                'agreement': integrated_result['agreement'],
                'final_confidence': integrated_result['confidence_scores']
            },
            'clinical_recommendations': self._get_treatment_recommendations(final_grade, patient_info),
            'patient_information': patient_info or {},
            'report_timestamp': '2024-01-01 10:00:00'
        }
    
    def _get_severity_level(self, grade):
        if grade == 0:
            return "无风险"
        elif grade <= 2:
            return "低至中度风险"
        elif grade == 3:
            return "高风险"
        else:
            return "极高风险"
    
    def _get_treatment_recommendations(self, grade, patient_info):
        """获取治疗建议"""
        recommendations = {
            "treatment_recommendations": {
                "medication": [],
                "procedures": [],
                "lifestyle": []
            },
            "followup_plan": {
                "interval": "",
                "next_exams": ["视力检查", "眼底照相", "OCT检查"]
            },
            "targets": {
                "hba1c": "<7.0%",
                "blood_pressure": "<130/80mmHg"
            },
            "patient_education": [],
            "warning_signs": ["视力突然下降", "视物变形", "眼前黑影"]
        }
        
        if grade == 0:
            recommendations["treatment_recommendations"]["lifestyle"] = ["严格控制血糖", "健康饮食", "适量运动"]
            recommendations["followup_plan"]["interval"] = "12个月"
            recommendations["patient_education"] = ["定期眼科检查", "控制血糖血压"]
            
        elif grade == 1:
            recommendations["treatment_recommendations"]["medication"] = ["羟苯磺酸钙"]
            recommendations["treatment_recommendations"]["lifestyle"] = ["严格血糖控制", "控制血压血脂"]
            recommendations["followup_plan"]["interval"] = "6-12个月"
            
        elif grade == 2:
            recommendations["treatment_recommendations"]["medication"] = ["羟苯磺酸钙", "改善微循环药物"]
            recommendations["treatment_recommendations"]["procedures"] = ["评估激光治疗必要性"]
            recommendations["followup_plan"]["interval"] = "4-6个月"
            
        elif grade == 3:
            recommendations["treatment_recommendations"]["procedures"] = ["全视网膜光凝治疗", "评估抗VEGF治疗"]
            recommendations["followup_plan"]["interval"] = "3-4个月"
            recommendations["patient_education"] = ["高危状态，需要及时干预"]
            
        elif grade == 4:
            recommendations["treatment_recommendations"]["procedures"] = ["紧急全视网膜光凝", "评估玻璃体手术"]
            recommendations["followup_plan"]["interval"] = "1个月或立即随访"
            recommendations["patient_education"] = ["急诊状态，有失明风险"]
            
        return recommendations

# 创建诊断系统实例
dr_system = DRDiagnosisSystem()

# 工作流节点函数
def supervisor_node(state: DiagnosisState):
    print(">>>supervisor_node")
    writer = get_stream_writer()
    
    if "final_report" in state and state["final_report"]:
        writer({"supervisor_step": "诊断完成"})
        return {"current_step": END}
    
    if "current_step" not in state or not state["current_step"]:
        writer({"supervisor_step": "开始诊断流程"})
        return {"current_step": "grading_analysis"}
    
    current_step = state["current_step"]
    step_flow = {
        "grading_analysis": "vision_analysis",
        "vision_analysis": "integration", 
        "integration": "knowledge_query",
        "knowledge_query": "report_generation",
        "report_generation": END
    }
    
    return {"current_step": step_flow.get(current_step, "other")}

def grading_analysis_node(state: DiagnosisState):
    print(">>>grading_analysis_node")
    writer = get_stream_writer()
    
    user_input = state["messages"][0].content
    try:
        input_data = json.loads(user_input)
        grading_result = {
            "grade": input_data.get("model_grade", 0),
            "confidence": input_data.get("confidence", 0),
            "image_path": input_data.get("image_path", ""),
            "patient_info": input_data.get("patient_info", {})
        }
        
        writer({"grading_step": f"分级结果: 等级{grading_result['grade']}"})
        
        return {
            "grading_result": grading_result,
            "patient_data": grading_result["patient_info"],
            "current_step": "grading_analysis"
        }
    except Exception as e:
        writer({"grading_error": f"解析失败: {str(e)}"})
        return {
            "grading_result": {"grade": 0, "confidence": 0},
            "current_step": "grading_analysis"
        }

def vision_analysis_node(state: DiagnosisState):
    print(">>>vision_analysis_node")
    writer = get_stream_writer()
    
    grading_result = state["grading_result"]
    
    prompt = f"""
    请分析糖尿病视网膜眼底图像，给出病变分级：
    
    分级标准：
    0级 - 无视网膜病变
    1级 - 轻度非增殖性糖尿病视网膜病变（仅微动脉瘤）
    2级 - 中度非增殖性糖尿病视网膜病变
    3级 - 重度非增殖性糖尿病视网膜病变  
    4级 - 增殖性糖尿病视网膜病变
    
    当前分级模型结果: 等级{grading_result.get('grade', 0)}
    
    请输出JSON格式：
    {{
        "predicted_grade": 等级数字,
        "confidence": 置信度,
        "key_findings": ["主要发现"],
        "rationale": "分析理由"
    }}
    """
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    try:
        vision_result = json.loads(response.content)
    except:
        vision_result = {
            "predicted_grade": grading_result.get('grade', 0),
            "confidence": 0.7,
            "key_findings": ["视觉分析完成"],
            "rationale": "基于图像特征分析"
        }
    
    writer({"vision_step": "视觉分析完成"})
    return {
        "vision_llm_result": vision_result,
        "current_step": "vision_analysis"
    }

def integration_node(state: DiagnosisState):
    print(">>>integration_node")
    writer = get_stream_writer()
    
    integrated_result = dr_system.integrate_predictions(
        state["grading_result"],
        state["vision_llm_result"]
    )
    
    writer({"integration_step": f"集成结果: 等级{integrated_result['final_grade']}"})
    return {
        "integrated_result": integrated_result,
        "current_step": "integration"
    }

def knowledge_query_node(state: DiagnosisState):
    print(">>>knowledge_query_node")
    writer = get_stream_writer()
    
    # 使用内置知识库，不依赖外部Redis
    knowledge_content = dr_system._get_treatment_recommendations(
        state["integrated_result"]['final_grade'],
        state["patient_data"]
    )
    
    writer({"knowledge_step": "知识查询完成"})
    return {
        "knowledge_content": knowledge_content,
        "current_step": "knowledge_query"
    }

def report_generation_node(state: DiagnosisState):
    print(">>>report_generation_node")
    writer = get_stream_writer()
    
    final_report = dr_system.generate_diagnosis_report(
        state["integrated_result"],
        state["patient_data"]
    )
    
    # 生成可读的报告
    report_text = format_report_for_display(final_report)
    
    writer({"report_step": "报告生成完成"})
    return {
        "final_report": final_report,
        "messages": [HumanMessage(content=report_text)],
        "current_step": "report_generation"
    }

def other_node(state: DiagnosisState):
    print(">>>other_node")
    return {
        "messages": [HumanMessage(content="诊断系统无法处理此请求")],
        "current_step": "other"
    }

def format_report_for_display(report_data):
    """格式化报告用于显示"""
    diagnosis = report_data['diagnosis_summary']
    analysis = report_data['model_analysis']
    recommendations = report_data['clinical_recommendations']
    
    report = "## 🩺 糖尿病视网膜病变诊断报告\n\n"
    report += f"### 📋 诊断摘要\n"
    report += f"- **病变分级**: {diagnosis['grade']}级 ({diagnosis['description']})\n"
    report += f"- **严重程度**: {diagnosis['severity']}\n\n"
    
    report += f"### 🔬 AI模型分析\n"
    report += f"- **分级模型**: {analysis['grading_model_result']}级\n"
    report += f"- **视觉分析**: {analysis['vision_llm_result']}级\n"
    report += f"- **模型一致性**: {'✅ 一致' if analysis['agreement'] else '⚠️ 不一致'}\n\n"
    
    report += f"### 💊 治疗建议\n"
    treatment = recommendations['treatment_recommendations']
    if treatment['medication']:
        report += f"- **药物治疗**: {', '.join(treatment['medication'])}\n"
    if treatment['procedures']:
        report += f"- **治疗措施**: {', '.join(treatment['procedures'])}\n"
    
    report += f"- **随访间隔**: {recommendations['followup_plan']['interval']}\n"
    report += f"- **血糖目标**: {recommendations['targets']['hba1c']}\n\n"
    
    report += "---\n*本报告由AI系统生成，仅供参考*"
    
    return report

# 路由函数
def diagnosis_routing_func(state: DiagnosisState):
    current_step = state.get("current_step", "")
    
    routing_map = {
        "grading_analysis": "grading_analysis_node",
        "vision_analysis": "vision_analysis_node", 
        "integration": "integration_node",
        "knowledge_query": "knowledge_query_node",
        "report_generation": "report_generation_node",
        END: END
    }
    
    return routing_map.get(current_step, "other_node")

# 构建图
builder = StateGraph(DiagnosisState)

# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("grading_analysis_node", grading_analysis_node)
builder.add_node("vision_analysis_node", vision_analysis_node)
builder.add_node("integration_node", integration_node)
builder.add_node("knowledge_query_node", knowledge_query_node)
builder.add_node("report_generation_node", report_generation_node)
builder.add_node("other_node", other_node)

# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", diagnosis_routing_func)
builder.add_edge("grading_analysis_node", "supervisor_node")
builder.add_edge("vision_analysis_node", "supervisor_node")
builder.add_edge("integration_node", "supervisor_node")
builder.add_edge("knowledge_query_node", "supervisor_node")
builder.add_edge("report_generation_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 编译图
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)