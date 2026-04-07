## 注：
- 从第二次实验开始作业均在这个库中提交，第一次作业在https://github.com/jnu-cv-lab/2023101136_automation_Liao_Junqian_homework1
- 第二次实验的实验报告（实验结果与分析）在homework2文件夹，学习报告内容（多个文件编译）在本README
- JSON file：第二次实验作业的配置文件
- debug_result：第二次实验作业的调试结果

## 实验二学习报告
- 如果需要多个文件进行编译：
- 把所有 .cpp 都写进 tasks.json 的 args 里，如：
- "args": [
-    "-g",
-    "main.cpp",        // 主文件
-    "project1.cpp",           // 文件1
-    "project2.cpp",           // 文件2
-    "-o",
-   ...
- ]
