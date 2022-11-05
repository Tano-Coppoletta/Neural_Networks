
rnd_images=[]
for i in range(9):
  with torch.no_grad():
    tensor=torch.rand((1,4))
    image  = decoder(tensor).detach().squeeze().numpy()
    rnd_images.append(image)

matrix_with_images=np.zeros((28*3,28*3))
img_num=0
for i in range(3):
  for j in range(3):
    for pos_x in range(28):
      for pos_y in range(28):
        matrix_with_images[i*28 + pos_x][j*28 + pos_y]=rnd_images[img_num][pos_x][pos_y]
    img_num+=1
plt.imshow(matrix_with_images, cmap='gist_gray')

# put your clustering accuracy calculation here
train_loader2 = torch.utils.data.DataLoader(train_data, batch_size=1)
output_of_encoder = []
labels=[]
for image,label in train_loader2:
  output = encoder(image).detach().squeeze().numpy()
  output_of_encoder.append(output)
  labels.append(label)

from sklearn.cluster import KMeans

output_of_encoder_numpy = np.array(output_of_encoder)
kmeans = KMeans(n_clusters=10).fit(output_of_encoder_numpy)

matrix = np.zeros((10,10))
for i in range(10):
  for j in range(10):
    for z in range(48000):
      if labels[z].item()==i and kmeans.labels_[z]==j:
        matrix[i][j]+=1

for i in range(10):
  max_index = np.argmax(matrix[i,:])
  map[i]=max_index
print(map)
i=0
j=0
while i<10:
  while j<10:
      if map[i]==map[j] and i!=j:
          if matrix[i][int(map[i])]>matrix[j][int(map[j])]:
              #reassign map[j]
              matrix[j][int(map[j])]=0
              map[j]=np.argmax(matrix[j,:])
              i=0
              j=0
      j+=1
  i+=1


print(map)
#compute accuracy
count=0
for i in range(10):
  for j in range(48000):
    if kmeans.labels_[j]==map[i] and labels[j].item()==i:
      count+=1
accuracy = count/48000 *100
print("The accuracy is: ",accuracy,"%")

