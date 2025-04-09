import streamlit as st

st.set_page_config(layout="wide")  # 宽屏模式
st.title("你的应用名称")

     # 侧边栏输入参数with st.sidebar:
param1 = st.slider("参数1", 0, 100, 50)
param2 = st.text_input("参数2", "默认值")

     # 主面板结果展示
if st.button("运行模型"):
         result = f"参数1={param1}, 参数2={param2}"
         st.success("运行成功！")
         st.subheader("结果")
         st.write(result)