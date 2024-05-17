# Example of processing tables in a PDF document using RAG

# 1. 现将每页PDF转成图片

# !pip install PyMuPDF
# !pip install timm
# !pip install --upgrade requests urllib3

import os
import fitz
from PIL import Image
import matplotlib.pyplot as plt

def pdf2images(pdf_file):
    '''将 PDF 每页转成一个 PNG 图像'''
    # 保存路径为原 PDF 文件名（不含扩展名）
    output_directory_path, _ = os.path.splitext(pdf_file)
    
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    
    # 加载 PDF 文件
    pdf_document = fitz.open(pdf_file)
    
    # 每页转一张图
    for page_number in range(pdf_document.page_count):
        # 取一页
        page = pdf_document[page_number]
    
        # 转图像
        pix = page.get_pixmap()
    
        # 从位图创建 PNG 对象
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
        # 保存 PNG 文件
        image.save(f"./{output_directory_path}/page_{page_number + 1}.png")
    
    # 关闭 PDF 文件
    pdf_document.close()

def show_images(dir_path):
    '''显示目录下的 PNG 图像'''
    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            # 打开图像
            img = Image.open(os.path.join(dir_path, file)) 

            # 显示图像
            plt.imshow(img)
            plt.axis('off')  # 不显示坐标轴
            plt.show()

pdf2images("llama2_page8.pdf")
show_images("llama2_page8")

# 2. 识别文档（图片）中的表格

class MaxResize(object):
    '''缩放图像'''
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image
    
import torchvision.transforms as transforms

# 图像预处理
detection_transform = transforms.Compose(
    [
        MaxResize(800),  # 将图像的最长边缩放至不超过800像素
        transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 使用均值和标准差对图像张量进行归一化
    ]
)

from transformers import AutoModelForObjectDetection

# 加载 TableTransformer 模型
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)

# 识别后的坐标换算与后处理
import torch
def box_cxcywh_to_xyxy(x):
    '''坐标转换'''
    x_c, y_c, w, h = x.unbind(-1)  # 将输入的张量x在最后一个维度上解绑，得到中心点坐标(x_c, y_c)和宽度w、高度h
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]  # 计算左上角和右下角的坐标
    return torch.stack(b, dim=1)  # 将计算得到的坐标堆叠起来，形成新的张量


def rescale_bboxes(out_bbox, size):
    '''区域缩放'''
    width, height = size  # 获取图像的宽度和高度
    boxes = box_cxcywh_to_xyxy(out_bbox)  # 将中心点坐标和宽高转换为左上角和右下角的坐标
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )  # 将坐标按照图像的宽度和高度进行缩放
    return boxes  # 返回缩放后的坐标


def outputs_to_objects(outputs, img_size, id2label):
    '''从模型输出中取定位框坐标'''
    m = outputs.logits.softmax(-1).max(-1)  # 对模型的输出进行softmax处理，并取得每个类别的最大概率值和对应的索引
    pred_labels = list(m.indices.detach().cpu().numpy())[0]  # 将预测的类别索引转换为numpy数组
    pred_scores = list(m.values.detach().cpu().numpy())[0]  # 将预测的类别概率转换为numpy数组
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]  # 获取预测的边界框坐标
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]  # 将边界框坐标按照图像的尺寸进行缩放，并转换为列表

    objects = []  # 初始化一个空列表，用于存储识别出的对象
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):  # 遍历每一个预测的类别、概率和边界框
        class_label = id2label[int(label)]  # 将类别索引转换为类别名称
        if not class_label == "no object":  # 如果识别出的不是"no object"
            objects.append(
                {
                    "label": class_label,  # 类别名称
                    "score": float(score),  # 类别概率
                    "bbox": [float(elem) for elem in bbox],  # 边界框坐标
                }
            )  # 将识别出的对象添加到列表中

    return objects  # 返回识别出的对象列表

# 识别表格，并将表格部分单独存为图像文件

def detect_and_crop_save_table(file_path):
    # 加载图像（PDF页）    
    image = Image.open(file_path)

    filename, _ = os.path.splitext(os.path.basename(file_path))

    # 输出路径
    cropped_table_directory = os.path.join(os.path.dirname(file_path), "table_images")

    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)

    # 预处理
    pixel_values = detection_transform(image).unsqueeze(0)

    # 识别表格
    with torch.no_grad():
        outputs = model(pixel_values)

    # 后处理，得到表格子区域
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    print(f"number of tables detected {len(detected_tables)}")

    for idx in range(len(detected_tables)):
        # 将识别从的表格区域单独存为图像
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        cropped_table.save(os.path.join(cropped_table_directory,f"{filename}_{idx}.png"))

detect_and_crop_save_table("llama2_page8/page_1.png")
show_images("llama2_page8/table_images")

# 3. 基于 GPT-4 Vision API 做表格问答
# 导入base64库，用于将图片转换为base64编码
import base64
# 导入openai库，用于调用OpenAI的API
from openai import OpenAI

# 导入dotenv库，用于加载环境变量
from dotenv import load_dotenv, find_dotenv
# 加载环境变量
_ = load_dotenv(find_dotenv())

# 创建OpenAI的客户端
client = OpenAI()

# 定义一个函数，用于将图片转换为base64编码
def encode_image(image_path):
  # 打开图片文件
  with open(image_path, "rb") as image_file:
    # 读取图片文件，将其转换为base64编码，并将结果解码为字符串
    return base64.b64encode(image_file.read()).decode('utf-8')

# 定义一个函数，用于对图片进行问答
def image_qa(query, image_path):
    # 将图片转换为base64编码
    base64_image = encode_image(image_path)
    # 调用OpenAI的API，传入问题和图片，获取回答
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        seed=42,
        messages=[{
            "role": "user",
              "content": [
                  {"type": "text", "text": query},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}",
                      },
                  },
              ],
        }],
    )

    # 返回回答的内容
    return response.choices[0].message.content

response = image_qa("哪个模型在AGI Eval数据集上表现最好。得分多少","llama2_page8/table_images/page_1_0.png")
print(response)

# 4. 用 GPT-4 Vision 生成表格（图像）描述，并向量化用于检索

import chromadb
from chromadb.config import Settings


class NewVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def add_images(self, image_paths):
        '''向 collection 中添加图像'''
        documents = [
            image_qa("请简要描述图片中的信息",image)
            for image in image_paths
        ]
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))],  # 每个文档的 id
            metadatas=[{"image": image} for image in image_paths] # 用 metadata 标记源图像路径
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results
    
def get_embeddings(texts, model="text-embedding-3-small",dimensions=None):# text-embedding-3-large
    '''Encapsulate OpenAI's Embedding model interface'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]
    
images = []
dir_path = "llama2_page8/table_images"
for file in os.listdir(dir_path):
    if file.endswith('.png'):
        # 打开图像
        images.append(os.path.join(dir_path, file))

new_db_connector = NewVectorDBConnector("table_demo",get_embeddings)
new_db_connector.add_images(images)

query  = "哪个模型在AGI Eval数据集上表现最好。得分多少"

results = new_db_connector.search(query, 1)
metadata = results["metadatas"][0]
print("====检索结果====")
print(metadata)
print("====回复====")
response = image_qa(query,metadata[0]["image"])
print(response)



