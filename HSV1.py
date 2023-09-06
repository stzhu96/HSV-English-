import cv2
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("HSV Feature Display"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload')
])

def analyze_image(contents):
    # 将上传的内容解码为图像格式
    encoded_image = contents.split(',')[1]
    decoded_image = cv2.imdecode(np.frombuffer(
        bytes(base64.b64decode(encoded_image)), np.uint8), -1)

    # 将RGB图像转换为HSV图像
    hsv_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2HSV)

    # 分别获取H、S、V三个通道的ndarray数据
    img_h = hsv_image[:,:,0]
    img_s = hsv_image[:,:,1]
    img_v = hsv_image[:,:,2]

    # 按H、S、V三个通道分别计算颜色直方图
    h_hist = cv2.calcHist([hsv_image],[0],None,[180],[0,180])
    s_hist = cv2.calcHist([hsv_image],[1],None,[256],[0,256])
    v_hist = cv2.calcHist([hsv_image],[2],None,[256],[0,256])
    m, dev = cv2.meanStdDev(hsv_image)  # 计算H、S、V三通道的均值和方差

    # 计算三个通道的均值和标准差
    h_mean, h_std = np.mean(h_hist), np.std(h_hist)
    s_mean, s_std = np.mean(s_hist), np.std(s_hist)
    v_mean, v_std = np.mean(v_hist), np.std(v_hist)
    m, dev = cv2.meanStdDev(hsv_image)

    return h_mean, h_std, s_mean, s_std, v_mean, v_std,m,dev

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output_image_upload(contents):
    if contents is not None:
        h_mean, h_std, s_mean, s_std, v_mean, v_std ,m,dev= analyze_image(contents)

        # 将图像显示在网页上
        return html.Div([
            html.H3('上传的图像：'),
            html.Img(src=contents, style={'width': '400px'}),
            #html.P(f'H通道均值：{h_mean:.2f}，标准差：{h_std:.2f}'),
            #html.P(f'S通道均值：{s_mean:.2f}，标准差：{s_std:.2f}'),
            #html.P(f'V通道均值：{v_mean:.2f}，标准差：{v_std:.2f}'),
            html.P(f'H mean：{m.ravel().tolist()[2]:.2f}，H standard deviation：{dev.ravel().tolist()[2]:.2f}'),
            html.P(f'S mean：{m.ravel().tolist()[1]:.2f}，S standard deviation：{dev.ravel().tolist()[1]:.2f}'),
            html.P(f'V mean：{m.ravel().tolist()[0]:.2f}，V standard deviation：{dev.ravel().tolist()[0]:.2f}'),
        ])


if __name__ == '__main__':
    app.run_server(debug=False)