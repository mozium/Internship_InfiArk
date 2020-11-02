# from pre_process import predict
import dash_core_components as dcc
import dash_html_components as html
import time


def anomaly_detect(y_real, y_pred, mape, mse, mae, thrshold):
    """
        根據預測結果去判斷是否異常。
            情況一: 差距 > 20%。
            情況二: metrics異常。

        Returns:
            dcc.Markdown
    """
    diff = []
    for i, (real, pred) in enumerate(zip(y_real, y_pred)):
        if real != 0:
            percent = (abs(real - pred) / real)
        else:
            if pred == 0:
                diff.append((i, False))
                continue
            else:
                percent = (abs(real - pred) / pred)
        if percent >= thrshold:
            diff.append((i, True))
        else:
            diff.append((i, False))
    
    time_str = time.asctime(time.localtime(time.time()))    # Thu Apr  7 10:05:21 2016



