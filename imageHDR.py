import numpy as np
import cv2

print("Hello worldl")
# 1. Bilder in Array Laden und Exposure times bestimmen
img1 = cv2.imread('images/window_exp_15_1.jpg')
img2 = cv2.imread('images/window_exp_4_1.jpg')
img3 = cv2.imread('images/window_exp_1_1.jpg')
img4 = cv2.imread('images/window_exp_1_4.jpg')
img5 = cv2.imread('images/window_exp_1_15.jpg')
img6 = cv2.imread('images/window_exp_1_60.jpg')
img7 = cv2.imread('images/window_exp_1_250.jpg')
img8 = cv2.imread('images/window_exp_1_1000.jpg')
img9 = cv2.imread('images/window_exp_1_4000.jpg')

images = [img1, img2, img3, img4, img5, img6, img7, img8, img9]
belichtungsZeit = np.array([15.0, 4.0, 1.0, 0.25, 0.066, 0.016, 0.004, 0.0001, 0.00025], dtype=np.float32)

# 2. Vereinigung aller Bilder zu einem HDR Bild mit Belichtungszeit
mDebvec = cv2.createMergeDebevec()
hdrDebvec = mDebvec.process(images, times=belichtungsZeit.copy())


# # alternative nach robertson mit Belichtungszeit
mRobertson = cv2.createMergeRobertson()
hdrRobertson = mRobertson.process(images, times=belichtungsZeit.copy())

# 3. Mappen der HDR daten in range [0...1]
map = cv2.createTonemapDurand(gamma=2.2)
resDebvec = map.process(hdrDebvec.copy())

# # Mappen von Robertson
map2 = cv2.createTonemapDurand(gamma=1.3)
resRobertson = map2.process(hdrRobertson.copy())

# 4. Alternativer Weg mit Mertens Algorithmus ohne Belichtungszeit
mergeMertens = cv2.createMergeMertens()
resMertens = mergeMertens.process(images)

# 5. Konvertiere zu 8-bit und speichere neue Bilder
confDebvec = np.clip(resDebvec * 255, 0, 255).astype('uint8')
confRobertson = np.clip(resRobertson * 255, 0, 255).astype('uint8')
confMertens = np.clip(resMertens * 255, 0, 255).astype('uint8')

# 6. Abspeichern der Images
cv2.imwrite("debvec.jpg", confDebvec)
cv2.imwrite("robertson.jpg", confRobertson)
cv2.imwrite("mertens.jpg", confMertens)

