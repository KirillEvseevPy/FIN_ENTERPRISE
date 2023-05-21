import sys
import os.path
import csv
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PyQt6.QtWidgets import QApplication,  QFileDialog, QMessageBox, QWidget, QGridLayout, QLineEdit, QPushButton,\
    QLabel, QProgressDialog, QProgressBar, QCheckBox
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision import transforms
import numpy as np
from PIL import Image

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Расспознавание лебедей')
        self.setGeometry(100, 100, 400, 100)


        self.prog_bar = QProgressBar(self)
        self.prog_bar.setGeometry(50, 50, 300, 30)
        self.prog_bar.setMinimum(0)

        layout = QGridLayout()
        self.setLayout(layout)

        # directory selection
        dir_btn = QPushButton('Выбрать')
        dir_btn.clicked.connect(self.open_dir_dialog)
        self.dir_name_edit = QLineEdit()

        start_btn = QPushButton('Запуск')
        start_btn.clicked.connect(self.swans_recognition)

        self.cbutton = QCheckBox("Режим энергосбережения")
        self.cbutton.setChecked(True)
        
        layout.addWidget(QLabel('Папка:'), 1, 0)
        layout.addWidget(self.dir_name_edit, 1, 1)
        layout.addWidget(dir_btn, 1, 2)
        layout.addWidget(start_btn, 2, 2)
        layout.addWidget(QLabel('Прогресс:'), 2, 0)
        layout.addWidget(self.prog_bar, 2, 1)
        layout.addWidget(self.cbutton, 4, 1)
         
        self.show()

    def open_dir_dialog(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if dir_name:
            path = Path(dir_name)
            self.dir_name_edit.setText(str(path))
            
            
    def swans_recognition(self):
        self.prog_bar.setValue(0)
        dir_photo = self.dir_name_edit.text()
        self.check_dir(dir_photo)
        total_file = sum(len(files) for address, dirs, files in os.walk(dir_photo))
        self.prog_bar.setMaximum(total_file-2)
        
        if self.cbutton.isChecked():
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(in_features=512, out_features=3)
            model.load_state_dict(torch.load("model.pt",map_location ='cpu'))
        else:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
            model.load_state_dict(torch.load("Final_model.pt",map_location={'cuda:0': 'cpu'}))          
        
        model.eval()
        
        convert_tensor = transforms.ToTensor()
        
        total_litle = 0
        total_klikun = 0
        total_shipun = 0
        
        with open('submission.csv', 'w', newline='') as csvfile:
            swans_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for address, dirs, files in os.walk(dir_photo):
                for name in files:

                    value = self.prog_bar.value()
                    self.prog_bar.setValue(value + 1)
                    full_path_img = os.path.join(address, name)

                    if self.cbutton.isChecked():

                        img = Image.open(full_path_img)
                        basewidth = 224
                        wpercent = (basewidth / float(img.size[0]))
                        hsize = int((float(img.size[1]) * float(wpercent)))
                        img_resize = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)

                        type_swan = str(torch.max(model(convert_tensor(img_resize).unsqueeze(0)),1)[1])[-3]
                        print(name)
                    else:
                        test_transform = A.Compose(
                            [
                                A.Resize(224, 224),
                                ToTensorV2(),

                            ]
                        )
                        pic = np.array(Image.open(full_path_img).convert("RGB"))
                        if test_transform is not None:
                            augmentations = test_transform(image = pic)
                            pic = augmentations['image'].to("cpu")
                            model.to("cpu")
                            type_swan = str(torch.max(model(pic.type(torch.float32).unsqueeze(0)), 1)[1])[-3]

                    # кликун - 0, малой -1 , шипун - 2 => Малый – 1 Кликун - 2 Шипун - 3
                    if type_swan == "0":
                        type_swan = "2"
                        total_klikun = total_klikun + 1
                    elif type_swan == "2":
                        type_swan = "3"
                        total_shipun = total_shipun + 1
                    else:
                        total_litle = total_litle + 1         
                        
                    swans_writer.writerow([name, type_swan])

            dlg_complite = QMessageBox(self)
            dlg_complite.setWindowTitle("Распознавание закончено!")
            dlg_complite.setText(f"Обработано {total_file} файлов! Обнаружены лебеди: Малый {total_litle}, Кликун {total_klikun}, Шипун {total_shipun}. Результаты сохранены в файл submission.csv")
            button = dlg_complite.exec()

    def check_dir(self, dir_photo):
        if not dir_photo:
            dlg_no_dir = QMessageBox(self)
            dlg_no_dir.setWindowTitle("Внимание!")
            dlg_no_dir.setText("Не указана папка с фотографиями!")
            button = dlg_no_dir.exec()
        elif not os.path.isdir(dir_photo):
            dlg_no_is_dir = QMessageBox(self)
            dlg_no_is_dir.setWindowTitle("Внимание!")
            dlg_no_is_dir.setText("Указаная папка не существует!")
            button = dlg_no_is_dir.exec()
        elif not os.listdir(dir_photo):
            dlg_no_listdir = QMessageBox(self)
            dlg_no_listdir.setWindowTitle("Внимание!")
            dlg_no_listdir.setText("В указаной папке нет файлов!")
            button = dlg_no_listdir.exec()
                  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
