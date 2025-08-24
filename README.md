# Langgraph_cooperative_multi_agent 项目

一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的高级工作流系统，实现了复杂的多步骤任务处理和状态管理。
<img width="1310" height="1248" alt="多agent互评互改" src="https://github.com/user-attachments/assets/92002049-4994-4887-be2f-c18ebac5a5c0" />


## 架构概述

本项目采用 LangGraph 的图形化工作流设计，多agent完成任务后，对其他agent完成的任务进行评价与改正。

## 视频演示



https://github.com/user-attachments/assets/09c58b7b-ab05-4825-a64b-0b7b4d29b764



## 项目结构

```
project/
├── src/                    # 源代码
│   └── agent
│       └── graphs.py            # 工作流图定义
```


## 贡献指南

我们欢迎社区贡献！请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 支持

如果您遇到问题或有疑问：

1. 查看 [文档](./README.md)
2. 提交 [GitHub Issue](https://github.com/DHU-JinQi/Langgraph_workflow_architecture/issues)



*此项目基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建，LangGraph 是 LangChain 生态系统的一部分。*
