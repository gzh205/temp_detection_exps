from tools.train import sendEmail

if __name__ == "__main__":
    sendEmail('1942592358@qq.com', filename='./tools/chart.html', aps=[[1, 2], [3, 4], [5,6]])
