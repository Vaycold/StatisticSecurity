# 모델에 패치이미지와 일반이미지를 둘다 넣자. 
gray_rec, perceptual_loss = model(aug_gray_batch, gray_batch)                
gray_rec2 = model2(aug_gray_batch)
