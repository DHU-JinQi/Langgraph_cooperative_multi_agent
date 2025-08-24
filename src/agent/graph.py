import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import logging
from typing import List, Literal, Optional, Dict, Any, Callable
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime

from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig

# ============= 日志配置 =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= 环境配置 =============
load_dotenv()
model = ChatDeepSeek(model="deepseek-chat", max_tokens=8000)

# ============= 数据模型定义 =============

class AgentFeedback(BaseModel):
    """Agent反馈模型"""
    agent_name: str = Field(description="评审Agent名称")
    feedback_type: str = Field(description="反馈类型：critique/suggestion/approval")
    feedback_content: str = Field(description="具体反馈内容")
    confidence_score: float = Field(description="置信度分数 0-1")
    suggested_improvements: List[str] = Field(description="改进建议")

from typing import Annotated
from langgraph.graph.message import add_messages

# 自定义的分析结果聚合函数
def add_analyses(existing: List[str], new: List[str]) -> List[str]:
    """聚合分析结果"""
    if existing is None:
        existing = []
    if isinstance(new, str):
        new = [new]
    return existing + new

# 添加完成状态跟踪
def add_completion_status(existing: Dict[str, bool], new: Dict[str, bool]) -> Dict[str, bool]:
    """跟踪各Agent完成状态"""
    if existing is None:
        existing = {}
    return {**existing, **new}

class MultiAgentState(TypedDict):
    """多Agent系统状态"""
    messages: Annotated[list, add_messages]
    original_query: Optional[str]
    analyses: Annotated[List[str], add_analyses]  # 改为可聚合的分析列表
    agent_feedbacks: Optional[List[AgentFeedback]]
    revision_count: Optional[int]
    consensus_reached: Optional[bool]
    final_report: Optional[str]
    workflow_stage: Optional[str]
    completion_status: Annotated[Dict[str, bool], add_completion_status]  # 新增完成状态跟踪

class FinancialAnalysisStep(BaseModel):
    step: str = Field(description="分析步骤名称")
    method: str = Field(description="使用的分析方法")
    data_needed: str = Field(description="此步骤需要的数据")
    assigned_agent: str = Field(description="负责的Agent")

class FinancialAnalysisPlan(BaseModel):
    analysis_steps: List[FinancialAnalysisStep]

# ============= 工具定义 =============

@tool
def get_stock_data(symbol: str, period: str = "1y") -> str:
    """获取股票基础数据（模拟实现）"""
    logger.info(f"获取股票数据: {symbol}, 周期: {period}")
    return f"""
    股票代码: {symbol}
    时间周期: {period}
    
    基础数据:
    - 当前价格: 125.50
    - 市值: 500亿
    - P/E比率: 18.5
    - P/B比率: 2.3
    - ROE: 15.2%
    - 52周高点: 145.20
    - 52周低点: 98.30
    
    近期表现:
    - 日涨跌幅: +2.1%
    - 周涨跌幅: +5.3%
    - 月涨跌幅: +12.8%
    """

@tool
def get_financial_news(keyword: str, days: int = 7) -> str:
    """获取金融新闻信息（模拟实现）"""
    logger.info(f"获取金融新闻: {keyword}, 天数: {days}")
    return f"""
    关键词: {keyword}
    时间范围: 最近{days}天
    
    主要新闻:
    1. 公司发布Q3财报，营收同比增长15%
    2. 获得重要政府订单，总价值约10亿元
    3. 董事会批准股份回购计划
    4. 分析师上调目标价至150元
    5. 行业政策利好，相关板块普涨
    """

@tool
def technical_analysis(symbol: str, indicator: str = "MA") -> str:
    """技术分析工具（模拟实现）"""
    logger.info(f"技术分析: {symbol}, 指标: {indicator}")
    return f"""
    技术指标分析 - {symbol}
    指标类型: {indicator}
    
    移动平均线:
    - MA5: 123.45 (支撑位)
    - MA20: 118.20 (强支撑)
    - MA60: 115.80 (长期趋势线)
    
    技术信号:
    - MACD: 金叉信号，多头排列
    - RSI: 65 (略偏强势区域)
    - 成交量: 较前期放大30%
    
    关键价位:
    - 支撑位: 120.00
    - 阻力位: 130.00
    """

@tool
def portfolio_optimization(assets: str, risk_level: str = "medium") -> str:
    """投资组合优化分析"""
    logger.info(f"组合优化分析: {assets}, 风险水平: {risk_level}")
    return f"""
    投资组合优化结果:
    资产类别: {assets}
    风险水平: {risk_level}
    
    建议配置:
    - 股票: 60% (蓝筹股40% + 成长股20%)
    - 债券: 30% (政府债券20% + 企业债10%)  
    - 现金: 10%
    
    预期收益: 8-12%
    最大回撤: 15%
    夏普比率: 1.2
    """

@tool
def risk_assessment(position_size: str, market_cap: str) -> str:
    """风险评估工具"""
    logger.info(f"风险评估: 持仓规模={position_size}, 市值={market_cap}")
    return f"""
    风险评估报告:
    持仓规模: {position_size}
    市值规模: {market_cap}
    
    风险指标:
    - VaR (95%): 单日最大损失2.5%
    - Beta系数: 1.2 (高于市场)
    - 流动性风险: 低
    - 信用风险: 中等
    - 行业集中度: 偏高
    
    风险建议: 适当分散投资，控制单一持仓比例
    """

# ============= 多Agent定义 =============

# 搜索工具
search_tool = TavilySearch(max_results=5, topic="general")
basic_tools = [get_stock_data, get_financial_news, technical_analysis, search_tool]
advanced_tools = basic_tools + [portfolio_optimization, risk_assessment]

# Agent 1: 基本面分析专家
FUNDAMENTAL_ANALYST_PROMPT = """
你是资深基本面分析专家，专注于公司财务分析、行业分析和价值评估。

核心职责:
1. 深入分析公司财务报表和关键指标
2. 评估公司商业模式和竞争优势
3. 研究行业趋势和市场地位
4. 提供基于内在价值的投资建议

分析要点:
- 盈利能力分析 (ROE, ROA, 毛利率等)
- 成长性分析 (营收增长、利润增长等)
- 偿债能力分析 (资产负债率、流动比率等)
- 估值分析 (PE, PB, PEG等)

请基于获取的数据进行专业的基本面分析，输出格式要求清晰美观，包含明确的分析结论。
"""

fundamental_agent = create_react_agent(
    model=model,
    prompt=FUNDAMENTAL_ANALYST_PROMPT,
    tools=basic_tools,
)

# Agent 2: 技术分析专家
TECHNICAL_ANALYST_PROMPT = """
你是专业技术分析师，精通图表分析、技术指标和市场趋势研判。

核心职责:
1. 价格走势和图表形态分析
2. 技术指标信号解读
3. 支撑阻力位判断
4. 买卖时机建议

分析要点:
- 趋势线和形态分析
- 移动平均线系统
- 动量指标 (RSI, MACD, KDJ)
- 成交量分析
- 关键价位识别

请基于技术指标提供专业的技术面分析，输出格式要求清晰美观，包含明确的操作建议。
"""

technical_agent = create_react_agent(
    model=model,
    prompt=TECHNICAL_ANALYST_PROMPT,
    tools=basic_tools,
)

# Agent 3: 风险管理专家
RISK_ANALYST_PROMPT = """
你是专业风险管理分析师，专注于投资风险识别、评估和控制建议。

核心职责:
1. 市场风险评估
2. 信用风险分析
3. 流动性风险评估
4. 投资组合风险管理建议

分析要点:
- VaR和压力测试
- Beta系数和波动率分析
- 相关性分析
- 风险分散建议
- 仓位管理策略

请基于风险管理理论提供专业的风险分析，输出格式要求清晰美观，包含具体的风险控制措施。
"""

risk_agent = create_react_agent(
    model=model,
    prompt=RISK_ANALYST_PROMPT,
    tools=advanced_tools,
)

# Agent 4: 高级综合分析师（负责综合和质量控制）
SENIOR_ANALYST_PROMPT = """
你是资深投资总监，负责综合各专业分析师的观点，进行质量控制和最终决策。

核心职责:
1. 整合各专业领域分析结果
2. 识别分析中的矛盾和不一致
3. 评估各分析的可靠性和逻辑性
4. 提供综合投资建议

质量控制要点:
- 分析逻辑的一致性
- 数据使用的准确性
- 结论的合理性
- 风险提示的充分性

请基于专业经验对其他分析师的工作进行评审和综合，输出格式要求专业美观，包含明确的投资评级。
"""

senior_agent = create_react_agent(
    model=model,
    prompt=SENIOR_ANALYST_PROMPT,
    tools=advanced_tools,
)

# ============= 输出格式化工具 =============

def format_analysis_output(title: str, content: str, agent_name: str) -> str:
    """格式化分析输出，使其更美观"""
    separator = "=" * 60
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    formatted_content = f"""
{separator}
📊 {title}
👨‍💼 分析师：{agent_name}
⏰ 分析时间：{timestamp}
{separator}

{content}

{separator}
"""
    return formatted_content

def format_review_output(reviews: List[str]) -> str:
    """格式化评审输出"""
    separator = "=" * 60
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    formatted_content = f"""
{separator}
🔍 同行评议阶段
⏰ 评议时间：{timestamp}
{separator}

"""
    
    for i, review in enumerate(reviews, 1):
        formatted_content += f"📝 评议 {i}:\n{review}\n\n"
    
    formatted_content += f"{separator}\n"
    return formatted_content

def format_final_report(content: str) -> str:
    """格式化最终报告"""
    separator = "=" * 80
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    formatted_content = f"""
{separator}
🎯 最终综合投资分析报告
📈 多Agent协作完成
⏰ 报告时间：{timestamp}
{separator}

{content}

{separator}
✅ 报告完成 - 基于多专家协作与同行评议
{separator}
"""
    return formatted_content

# ============= 多Agent节点函数 =============

def coordinator_node(state: MultiAgentState) -> Dict[str, Any]:
    """协调器节点 - 分配任务给各专业Agent"""
    # 从消息中提取用户查询
    user_query = ""
    if state.get("original_query"):
        user_query = state["original_query"]
    elif state.get("messages") and len(state["messages"]) > 0:
        user_query = state["messages"][0].content
    
    logger.info(f"🚀 协调器启动 - 开始多Agent协作分析任务: {user_query}")
    
    start_message = format_analysis_output(
        "多Agent协作分析启动",
        f"任务内容：{user_query}\n正在调度基本面分析师、技术分析师、风险分析师进行并行分析...",
        "系统协调器"
    )
    
    return {
        "messages": [AIMessage(content=start_message)],
        "original_query": user_query,
        "workflow_stage": "coordination_started",
        "completion_status": {}  # 初始化完成状态
    }

def fundamental_analysis_node(state: MultiAgentState) -> Dict[str, Any]:
    """基本面分析节点"""
    # 安全获取查询内容
    query = state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][0].content
    
    logger.info("📊 基本面分析师开始工作")
    
    task = f"请对以下投资标的进行深入的基本面分析: {query}"
    
    try:
        result = fundamental_agent.invoke({"messages": [HumanMessage(content=task)]})
        analysis_content = result["messages"][-1].content
        
        formatted_output = format_analysis_output(
            "基本面分析报告",
            analysis_content,
            "基本面分析专家"
        )
        
        logger.info("✅ 基本面分析完成")
        
        return {
            "messages": [AIMessage(content=formatted_output)],
            "analyses": [f"基本面分析: {analysis_content}"],
            "completion_status": {"fundamental": True}
        }
        
    except Exception as e:
        logger.error(f"❌ 基本面分析失败: {e}")
        error_msg = format_analysis_output(
            "基本面分析报告",
            f"分析过程中出现错误: {str(e)}",
            "基本面分析专家"
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "analyses": [f"基本面分析: 分析失败 - {str(e)}"],
            "completion_status": {"fundamental": False}
        }

def technical_analysis_node(state: MultiAgentState) -> Dict[str, Any]:
    """技术分析节点"""
    # 安全获取查询内容
    query = state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][0].content
    
    logger.info("📈 技术分析师开始工作")
    
    task = f"请对以下投资标的进行专业的技术面分析: {query}"
    
    try:
        result = technical_agent.invoke({"messages": [HumanMessage(content=task)]})
        analysis_content = result["messages"][-1].content
        
        formatted_output = format_analysis_output(
            "技术分析报告",
            analysis_content,
            "技术分析专家"
        )
        
        logger.info("✅ 技术分析完成")
        
        return {
            "messages": [AIMessage(content=formatted_output)],
            "analyses": [f"技术分析: {analysis_content}"],
            "completion_status": {"technical": True}
        }
        
    except Exception as e:
        logger.error(f"❌ 技术分析失败: {e}")
        error_msg = format_analysis_output(
            "技术分析报告",
            f"分析过程中出现错误: {str(e)}",
            "技术分析专家"
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "analyses": [f"技术分析: 分析失败 - {str(e)}"],
            "completion_status": {"technical": False}
        }

def risk_analysis_node(state: MultiAgentState) -> Dict[str, Any]:
    """风险分析节点"""
    # 安全获取查询内容
    query = state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][0].content
    
    logger.info("⚠️ 风险分析师开始工作")
    
    task = f"请对以下投资标的进行全面的风险评估: {query}"
    
    try:
        result = risk_agent.invoke({"messages": [HumanMessage(content=task)]})
        analysis_content = result["messages"][-1].content
        
        formatted_output = format_analysis_output(
            "风险评估报告",
            analysis_content,
            "风险管理专家"
        )
        
        logger.info("✅ 风险分析完成")
        
        return {
            "messages": [AIMessage(content=formatted_output)],
            "analyses": [f"风险分析: {analysis_content}"],
            "completion_status": {"risk": True}
        }
        
    except Exception as e:
        logger.error(f"❌ 风险分析失败: {e}")
        error_msg = format_analysis_output(
            "风险评估报告",
            f"分析过程中出现错误: {str(e)}",
            "风险管理专家"
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "analyses": [f"风险分析: 分析失败 - {str(e)}"],
            "completion_status": {"risk": False}
        }

def wait_for_analyses_node(state: MultiAgentState) -> Dict[str, Any]:
    """等待所有分析完成的汇聚节点"""
    completion_status = state.get("completion_status", {})
    completed = [completion_status.get("fundamental", False), 
                completion_status.get("technical", False), 
                completion_status.get("risk", False)]
    
    logger.info(f"📋 分析完成状态检查: 基本面={completion_status.get('fundamental', False)}, "
                f"技术面={completion_status.get('technical', False)}, "
                f"风险={completion_status.get('risk', False)}")
    
    if all(completed):
        logger.info("✅ 所有专业分析已完成，准备进入同行评议阶段")
        return {
            "workflow_stage": "all_analyses_completed",
            "messages": [AIMessage(content="📝 所有专业分析已完成，正在准备同行评议...")]
        }
    else:
        logger.warning("⏳ 部分分析尚未完成，继续等待...")
        return {
            "workflow_stage": "waiting_for_analyses"
        }

def peer_review_node(state: MultiAgentState) -> Dict[str, Any]:
    """同行评议节点 - Agent互相评审"""
    
    logger.info("🔍 开始同行评议阶段 - Agent互评互改")
    
    # 从analyses字段收集所有分析结果
    analyses = state.get("analyses", [])
    combined_analysis = "\n\n".join(analyses)
    
    logger.info(f"📊 收集到的分析结果数量: {len(analyses)}")
    
    if not combined_analysis:
        logger.warning("⚠️ 没有找到分析结果，跳过同行评议")
        return {
            "messages": [AIMessage(content="【同行评议】没有分析结果可供评议")],
            "workflow_stage": "peer_review_completed"
        }
    
    # 让每个Agent评审其他Agent的工作
    feedbacks = []
    
    # 基本面分析师评审技术和风险分析
    fundamental_review_task = f"""
    作为基本面分析专家，请评审以下技术分析和风险分析的质量：
    
    {combined_analysis}
    
    请重点关注：
    1. 分析逻辑是否合理
    2. 是否与基本面分析结果一致
    3. 有哪些遗漏或错误
    4. 提出具体改进建议
    
    格式：【基本面分析师评审】
    评审意见：...
    改进建议：...
    """
    
    try:
        logger.info("👨‍💼 基本面分析师开始评审其他分析")
        fundamental_review = fundamental_agent.invoke({"messages": [HumanMessage(content=fundamental_review_task)]})
        feedbacks.append(f"【基本面分析师评审】\n{fundamental_review['messages'][-1].content}")
        logger.info("✅ 基本面分析师评审完成")
    except Exception as e:
        logger.error(f"❌ 基本面分析师评审失败: {e}")
        feedbacks.append("【基本面分析师评审】评审过程中出现错误")
    
    # 技术分析师评审基本面和风险分析
    technical_review_task = f"""
    作为技术分析专家，请评审以下基本面分析和风险分析的质量：
    
    {combined_analysis}
    
    请重点关注：
    1. 分析是否结合了市场技术面情况
    2. 时机判断是否合理
    3. 价格目标是否符合技术面支撑
    4. 提出具体改进建议
    
    格式：【技术分析师评审】
    评审意见：...
    改进建议：...
    """
    
    try:
        logger.info("👨‍💻 技术分析师开始评审其他分析")
        technical_review = technical_agent.invoke({"messages": [HumanMessage(content=technical_review_task)]})
        feedbacks.append(f"【技术分析师评审】\n{technical_review['messages'][-1].content}")
        logger.info("✅ 技术分析师评审完成")
    except Exception as e:
        logger.error(f"❌ 技术分析师评审失败: {e}")
        feedbacks.append("【技术分析师评审】评审过程中出现错误")
    
    # 风险分析师评审基本面和技术分析
    risk_review_task = f"""
    作为风险管理专家，请评审以下基本面分析和技术分析的质量：
    
    {combined_analysis}
    
    请重点关注：
    1. 风险因素是否被充分识别
    2. 风险评估是否客观准确
    3. 风险控制建议是否实用
    4. 提出具体改进建议
    
    格式：【风险分析师评审】
    评审意见：...
    改进建议：...
    """
    
    try:
        logger.info("👨‍⚖️ 风险分析师开始评审其他分析")
        risk_review = risk_agent.invoke({"messages": [HumanMessage(content=risk_review_task)]})
        feedbacks.append(f"【风险分析师评审】\n{risk_review['messages'][-1].content}")
        logger.info("✅ 风险分析师评审完成")
    except Exception as e:
        logger.error(f"❌ 风险分析师评审失败: {e}")
        feedbacks.append("【风险分析师评审】评审过程中出现错误")
    
    formatted_feedback = format_review_output(feedbacks)
    
    logger.info("🎯 同行评议阶段完成，共收集到 {} 条评审意见".format(len(feedbacks)))
    
    return {
        "messages": [AIMessage(content=formatted_feedback)],
        "agent_feedbacks": [
            AgentFeedback(
                agent_name="peer_review",
                feedback_type="critique", 
                feedback_content=formatted_feedback,
                confidence_score=0.8,
                suggested_improvements=["基于评议结果优化分析"]
            )
        ],
        "workflow_stage": "peer_review_completed"
    }

def senior_synthesis_node(state: MultiAgentState) -> Dict[str, Any]:
    """高级综合分析节点"""
    
    logger.info("🎯 高级投资总监开始综合分析和质量控制")
    
    # 收集所有分析和评议结果
    all_content = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            all_content.append(msg.content)
    
    combined_content = "\n\n".join(all_content)
    
    synthesis_task = f"""
    作为资深投资总监，请基于以下专业分析师的工作成果和同行评议结果，
    形成最终的综合投资分析报告：
    
    {combined_content}
    
    请提供：
    1. 📋 执行摘要
    2. 🎯 综合投资建议
    3. ⚠️ 关键风险提示
    4. 📊 投资评级和目标价位
    5. 🔍 各专业分析的整合和质量评估
    
    要求：
    - 逻辑清晰，结论明确
    - 充分考虑各方意见
    - 识别并解决分析中的矛盾
    - 提供实用的投资指导
    - 格式美观，条理清晰
    """
    
    try:
        result = senior_agent.invoke({"messages": [HumanMessage(content=synthesis_task)]})
        final_report_content = result["messages"][-1].content
        
        formatted_final_report = format_final_report(final_report_content)
        
        logger.info("✅ 最终综合报告生成完成")
        
        return {
            "messages": [AIMessage(content=formatted_final_report)],
            "final_report": final_report_content,
            "consensus_reached": True,
            "workflow_stage": "synthesis_completed"
        }
        
    except Exception as e:
        logger.error(f"❌ 综合分析失败: {e}")
        error_report = format_final_report(f"综合分析过程中出现错误: {str(e)}")
        return {
            "messages": [AIMessage(content=error_report)],
            "final_report": f"综合分析失败: {str(e)}",
            "consensus_reached": True,  # 即使失败也结束流程
            "workflow_stage": "synthesis_failed"
        }

def consensus_check_node(state: MultiAgentState) -> Dict[str, Any]:
    """共识检查节点"""
    
    logger.info("🔍 检查Agent共识状态")
    
    revision_count = state.get("revision_count", 0)
    
    # 简单的共识检查逻辑
    if revision_count < 2:  # 最多允许2轮修订
        # 检查是否需要进一步修订
        final_report = state.get("final_report", "")
        
        if len(final_report) > 500:  # 简单的质量检查
            consensus_reached = True
            logger.info("✅ 共识达成 - 报告质量满足要求")
        else:
            consensus_reached = False
            revision_count += 1
            logger.info(f"⏳ 需要继续修订 - 第{revision_count}轮修订")
    else:
        consensus_reached = True  # 超过修订次数限制，强制达成共识
        logger.info("✅ 共识达成 - 已达到最大修订次数")
    
    return {
        "consensus_reached": consensus_reached,
        "revision_count": revision_count,
        "workflow_stage": "consensus_checked"
    }

# ============= 路由条件函数 =============

def check_consensus_routing(state: MultiAgentState) -> str:
    """检查是否达成共识"""
    if state.get("consensus_reached", False):
        logger.info("🎯 工作流程完成，准备输出最终结果")
        return END
    else:
        logger.info("🔄 未达成共识，继续修订流程")
        return "peer_review"  # 继续修订流程

def check_analyses_completion(state: MultiAgentState) -> str:
    """检查所有分析是否完成"""
    completion_status = state.get("completion_status", {})
    completed = [completion_status.get("fundamental", False), 
                completion_status.get("technical", False), 
                completion_status.get("risk", False)]
    
    if all(completed):
        logger.info("✅ 所有分析已完成，进入同行评议")
        return "peer_review"
    else:
        missing = []
        if not completion_status.get("fundamental", False):
            missing.append("基本面分析")
        if not completion_status.get("technical", False):
            missing.append("技术分析") 
        if not completion_status.get("risk", False):
            missing.append("风险分析")
        logger.info(f"⏳ 等待分析完成，缺少: {', '.join(missing)}")
        return "wait_for_analyses"

# ============= 构建多Agent工作流图 =============

def build_multi_agent_graph():
    """构建多Agent协作工作流图"""
    builder = StateGraph(MultiAgentState)
    
    # 添加节点
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("fundamental_analysis", fundamental_analysis_node)
    builder.add_node("technical_analysis", technical_analysis_node) 
    builder.add_node("risk_analysis", risk_analysis_node)
    builder.add_node("wait_for_analyses", wait_for_analyses_node)  # 新增汇聚节点
    builder.add_node("peer_review", peer_review_node)
    builder.add_node("senior_synthesis", senior_synthesis_node)
    builder.add_node("consensus_check", consensus_check_node)
    
    # 设置入口点
    builder.add_edge(START, "coordinator")
    
    # 修复：使用正确的并行边设置
    # 协调器完成后，启动三个并行的分析任务
    builder.add_edge("coordinator", "fundamental_analysis")
    builder.add_edge("coordinator", "technical_analysis")
    builder.add_edge("coordinator", "risk_analysis")
    
    # 所有分析完成后都流向等待节点
    builder.add_edge("fundamental_analysis", "wait_for_analyses")
    builder.add_edge("technical_analysis", "wait_for_analyses") 
    builder.add_edge("risk_analysis", "wait_for_analyses")
    
    # 等待节点根据完成状态决定下一步
    builder.add_conditional_edges(
        "wait_for_analyses",
        check_analyses_completion,
        {
            "peer_review": "peer_review",
            "wait_for_analyses": "wait_for_analyses"  # 继续等待
        }
    )
    
    # 评议后进行高级综合
    builder.add_edge("peer_review", "senior_synthesis")
    
    # 综合后检查共识
    builder.add_edge("senior_synthesis", "consensus_check")
    
    # 条件边：根据共识情况决定是否结束
    builder.add_conditional_edges(
        "consensus_check",
        check_consensus_routing,
        {
            END: END,
            "peer_review": "peer_review"  # 未达成共识时继续评议
        }
    )
    
    return builder.compile()


# sample_query = "请分析腾讯控股(0700.HK)的投资价值，我想了解其基本面、技术面以及风险评估"

# 构建图实例供外部调用
graph = build_multi_agent_graph()