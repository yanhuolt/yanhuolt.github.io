from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from spire.doc import Document, FileFormat
import os
import logging
import uuid
import threading
import base64
import json
from typing import Dict, Any, Optional
from word_generator import convert_markdown_to_word_with_charts
from markdown_processor import process_markdown_for_charts

app = FastAPI(title="Enhanced Markdown to Word Converter with Charts")

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 文件存储配置
output_dir = "/var/www/files"
temp_dir = "/tmp/md_temp"
chart_dir = "/tmp/charts"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(chart_dir, exist_ok=True)
os.chmod(output_dir, 0o775)

# 全局服务地址配置
SERVER_HOST = "192.168.0.109"  # 替换为实际IP
PORT = 8089

# 并发控制锁
lock = threading.Lock()

@app.post("/office/word/convert")
async def convert_md_to_docx(request: Request):
    """原始的markdown转word接口（保持兼容性）"""
    logger.info('收到转换请求')

    content = await request.body()
    if not content:
        return JSONResponse({"error": "没有提供内容"}, status_code=400)

    try:
        content = content.decode('utf-8')
    except Exception as e:
        logger.error(f"解码错误: {str(e)}")
        return JSONResponse({"error": "内容编码错误"}, status_code=400)

    # 创建临时Markdown文件
    md_file_name = f"temp_{uuid.uuid4().hex}.md"
    md_file_path = os.path.join(temp_dir, md_file_name)

    try:
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logger.error(f"文件写入错误: {str(e)}")
        return JSONResponse({"error": "文件系统错误"}, status_code=500)

    # 使用锁避免并发冲突
    with lock:
        # 生成唯一文件名
        file_name = f"doc_{base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')}.docx"
        output_path = os.path.join(output_dir, file_name)

        try:
            doc = Document()
            doc.LoadFromFile(md_file_path, FileFormat.Markdown)
            doc.SaveToFile(output_path, FileFormat.Docx)
            doc.Dispose()
            logger.info(f"成功生成文件: {file_name}")
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            return JSONResponse({"error": "文档转换失败"}, status_code=500)
        finally:
            # 清理临时文件
            if os.path.exists(md_file_path):
                os.remove(md_file_path)

    # 生成纯净的下载链接（无JSON包裹）
    download_url = f"http://{SERVER_HOST}:{PORT}/office/word/download/{file_name}"

    # 返回纯文本响应（核心修复）
    return Response(
        content=download_url,
        media_type="text/plain",
        status_code=200
    )

@app.post("/office/word/convert_with_charts")
async def convert_md_to_docx_with_charts(request: Request):
    """增强版markdown转word接口，支持图表插入"""
    logger.info('收到带图表的转换请求')

    try:
        # 解析请求体
        body = await request.body()
        if not body:
            return JSONResponse({"error": "没有提供内容"}, status_code=400)

        # 尝试解析JSON格式的请求
        try:
            request_data = json.loads(body.decode('utf-8'))
            markdown_content = request_data.get('markdown', '')
            chart_configs = request_data.get('charts', {})
        except json.JSONDecodeError:
            # 如果不是JSON，则当作纯markdown内容处理
            markdown_content = body.decode('utf-8')
            chart_configs = {}

        if not markdown_content:
            return JSONResponse({"error": "没有提供markdown内容"}, status_code=400)

        # 使用锁避免并发冲突
        with lock:
            # 生成唯一文件名
            file_name = f"doc_{base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')}.docx"

            try:
                # 使用增强版转换器
                output_path = convert_markdown_to_word_with_charts(
                    markdown_content=markdown_content,
                    chart_configs=chart_configs,
                    output_dir=output_dir,
                    filename=file_name
                )

                logger.info(f"成功生成带图表的文件: {file_name}")

            except Exception as e:
                logger.error(f"转换失败: {str(e)}")
                return JSONResponse({"error": f"文档转换失败: {str(e)}"}, status_code=500)

        # 生成下载链接
        download_url = f"http://{SERVER_HOST}:{PORT}/office/word/download/{file_name}"

        # 返回纯文本响应
        return Response(
            content=download_url,
            media_type="text/plain",
            status_code=200
        )

    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        return JSONResponse({"error": f"处理请求失败: {str(e)}"}, status_code=500)

@app.get("/office/word/download/{filename}")
async def download_file(filename: str):
    # 安全验证
    if not filename.endswith(".docx") or "/" in filename or "\\" in filename:
        logger.error(f"无效文件名格式: {filename}")
        return HTTPException(400, "无效文件名格式")
    
    # 文件路径
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        # 尝试兼容可能的URL编码
        decoded_filename = filename.replace(" ", "_").replace("%20", "_")
        decoded_path = os.path.join(output_dir, decoded_filename)
        
        if os.path.exists(decoded_path):
            file_path = decoded_path
        else:
            existing_files = os.listdir(output_dir)
            logger.error(f"文件不存在! 目录内容: {', '.join(existing_files)}")
            return HTTPException(404, "文件未找到")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        access_log=True,
        timeout_keep_alive=30
    )