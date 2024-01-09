import streamlit as st
import os
import cv2
import urllib.request
from fastai.test_utils import *
from fastai.learner import load_learner
from fastai.vision import *
from torch import nn
import torch.nn.functional as F
from PIL import Image

base_loss = F.l1_loss


@st.cache_data
def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


@st.cache_data
def traditional_transform(img: Image):
    # 0.参数设置
    ksize = 15
    sigma = 50
    # 1.读取图片
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # 2.图片灰度处理
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # 3.图像取反
    img_inv = 255 - img_gray
    # 4.高斯滤波
    blur = cv2.GaussianBlur(img_inv, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    # 5.颜色减淡混合
    img_mix = cv2.divide(img_gray, 255 - blur, scale=255)
    # 6.图片展示
    return Image.fromarray(cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB))


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


@st.cache_data
def nn_transform(img: Image):
    im_temp = add_margin(img, 250, 250, 250, 250, (255, 255, 255))
    im_temp.save("temp.jpg", quality=95)
    image = open_image("temp.jpg")
    p, img_nn, b = learn.predict(image)
    return Image.fromarray(torch.clamp(img_nn.data * 255, 0, 255).byte().permute(1, 2, 0).numpy())


st.set_page_config(page_title="素描风格图像生成工具")

st.title("素描风格图像生成工具")
st.write("Made by brick and denny, grade 21, school of CSE, SCUT.")


@st.cache_resource
def get_model():
    MODEL_NAME = "ArtLine_920.pkl"
    # MODEL_NAME = "ArtLine_650.pkl"
    MODEL_URL = "https://www.dropbox.com/s/04suaimdpru76h3/ArtLine_920.pkl?dl=1"
    # MODEL_URL = "https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1"
    if not os.path.exists(MODEL_NAME):
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
    return load_learner(Path("."), MODEL_NAME)


@st.cache_resource
def get_extensions():
    extensions = Image.registered_extensions().items()
    return [f for ex, f in extensions if f in Image.OPEN]


with st.spinner("正在加载中，请稍等……"):
    learn = get_model()
    supported_extensions = get_extensions()

uploaded_file = st.file_uploader(
    "请上传一张待处理的图像:",
    type=supported_extensions,
    # type=["bmp"],
)

col1, col2, col3 = st.columns(3)

print_no_image_error = False
print_nn_info = False

with col1:
    if uploaded_file is not None:
        image_origin = Image.open(uploaded_file).convert("RGB")
        file_name = uploaded_file.name.rsplit('.', 1)[0]
        file_type = uploaded_file.name.rsplit('.', 1)[1]
        st.image(image_origin, caption="原图", use_column_width=True)
    if st.button("点我生成", use_container_width=True):
        if uploaded_file is not None and image_origin is not None:
            files = {"file": uploaded_file.getvalue()}
            with col2:
                with st.spinner("图像正在使用传统算法生成中，请稍等……"):
                    image_2 = traditional_transform(image_origin)
                st.image(image_2, caption="传统方法处理结果", use_column_width=True)
                buf_2 = BytesIO()
                image_2.save(buf_2, format="png")
                byte_im_2 = buf_2.getvalue()
                st.download_button(
                    label="保存这个结果",
                    data=byte_im_2,
                    file_name=f"{file_name}-传统方法.png",
                    mime=f"image/png",
                    use_container_width=True
                )

            with col3:
                with st.spinner("图像正在使用神经网络生成中，请稍等……"):
                    image_3 = nn_transform(image_origin)
                st.image(image_3, caption="神经网络处理结果", use_column_width=True)
                buf_3 = BytesIO()
                image_3.save(buf_3, format="png")
                byte_im_3 = buf_3.getvalue()
                st.download_button(
                    label="保存这个结果",
                    data=byte_im_3,
                    file_name=f"{file_name}-神经网络.png",
                    mime=f"image/png",
                    use_container_width=True
                )
                print_nn_info = True
        else:
            print_no_image_error = True

if print_no_image_error:
    st.error("不要着急，你需要先上传一张图片！")
if print_nn_info:
    st.info("神经网络生成图像的不确定性可能会导致图像边缘有额外的白色区域。如有需要，可下载图像后自行截去。", icon="ℹ️")
    st.info(
        "由于 streamlit 的机制，点击任意保存按钮后页面会刷新。针对这种情况我们进行了生成结果缓存处理，你只需再次点击生成按钮即可快速看到之前已经计算生成的结果。",
        icon="ℹ️")
