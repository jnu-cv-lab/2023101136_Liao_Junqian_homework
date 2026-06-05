## 注：
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
