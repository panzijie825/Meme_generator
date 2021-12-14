import simpletransformers.classification
from simpletransformers.classification import ClassificationModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import pickle

import torch
import torch.nn as nn
import torchvision

import numpy as np
from PIL import ImageFont, Image, ImageDraw
import torch.nn.functional as F

templates_dict = pickle.load( open( "model/templates_dict.pickle", "rb" ) ) 
temp_num2id = pickle.load( open( "model/temp_num2id.pickle", "rb" ) )
p_in = open('model/glove_6b.pickle','rb')
glove = pickle.load(p_in)
p_in.close()

p_in = open('model/id_to_vocab.pickle','rb')
int_to_vocab = pickle.load(p_in)
p_in.close()

p_in = open('model/vocab_to_id.pickle','rb')
vocab_to_int = pickle.load(p_in)
p_in.close()

weights = torch.zeros(len(int_to_vocab),100)
for i in range(len(weights)):
    word = int_to_vocab[i]
    if word in glove.keys():
        weights[i] = torch.from_numpy(glove[word])
    else:
        weights[i] = torch.from_numpy(np.random.randn(1,100))

class Encoder(nn.Module):
    
    def __init__(self,encoded_image_size = 14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        resnet = torchvision.models.resnet101(pretrained=True)
        
        modules = list(resnet.children())[:-2]
        
        self.resnet = nn.Sequential(*modules)
        self.adaptie_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
        
    def forward(self,images):
        out = self.resnet(images)
        out = self.adaptie_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, enc_img, hidden_state):
        enc_att = self.encoder_att(enc_img)     # (batch_size, num_pixels, attention_dim) 
        dec_att = self.decoder_att(hidden_state)   # (batch_size, attention_dim)
        a = self.full_att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(a)
        
        context = torch.sum(enc_img * alpha.unsqueeze(2), 1)
        return context,alpha   

        
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_dim, encoder_dim=2048, dropout=0.5):
        super(Decoder,self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim = hidden_dim, attention_dim=attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell((embedding_dim+encoder_dim), hidden_dim,bias=True)
        self.init_h = nn.Linear(encoder_dim,hidden_dim)
        self.init_c = nn.Linear(encoder_dim,hidden_dim)
        self.beta = nn.Linear(hidden_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)
    
    def load_glove(self,embeddings,require_grad=True):
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = require_grad
    
    def init_hidden_state(self, a):
        mean_a = a.mean(dim=1)
        h = self.init_h(mean_a)
        c = self.init_c(mean_a)
        return h,c
    
    def forward(self,enc_out,enc_captions,lengths):
        batch_size = enc_out.size(0)
        enc_dim = enc_out.size(-1)
        vocab_size = self.vocab_size
        
        enc_out = enc_out.view(batch_size, -1, enc_dim)
        num_pixels = enc_out.size(1)
        
        caption_lengths, ind = lengths.sort(dim=0,descending=True)
        enc_out = enc_out[ind]
        enc_captions = enc_captions[ind]

        
        embeddings = self.embedding(enc_captions)
        h,c = self.init_hidden_state(enc_out)
        decode_lengths = (caption_lengths - 1).tolist()
        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l>t for l in decode_lengths])
            attention_weighed_encoding, alpha = self.attention(enc_out[:batch_size_t],h[:batch_size_t])
            
            gate = self.sigmoid(self.beta(h[:batch_size_t]))
            attention_weighed_encoding = gate*attention_weighed_encoding
            
            h,c = self.lstm(torch.cat([embeddings[:batch_size_t,t,:], attention_weighed_encoding], dim=1),
                           (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t,t,:] = preds
            alphas[:batch_size_t,t,:] = alpha
        return predictions,alphas,enc_captions,decode_lengths,ind

def make_meme(topString, bottomString, img):
	#img = Image.open(filename)
  imageSize = img.shape
  #print(imageSize)

	# find font size that works
  fontSize = int(imageSize[1]/5)
  font = ImageFont.truetype("impact.ttf", fontSize)
  topTextSize = font.getsize(topString)
  bottomTextSize = font.getsize(bottomString)

  #decrease font size until fits
  while topTextSize[0] > imageSize[0]-20 or bottomTextSize[0] > imageSize[0]-20:
    fontSize = fontSize - 1
    font = ImageFont.truetype("impact.ttf", fontSize)
    topTextSize = font.getsize(topString)
    bottomTextSize = font.getsize(bottomString)

  # find top centered position for top text
  topTextPositionX = (imageSize[0]/2) - (topTextSize[0]/2)
  topTextPositionY = imageSize[1]/10
  topTextPosition = (topTextPositionX, topTextPositionY)

  # find bottom centered position for bottom text
  bottomTextPositionX = (imageSize[0]/2) - (bottomTextSize[0]/2)
  bottomTextPositionY = imageSize[1] - bottomTextSize[1]*1.5
  bottomTextPosition = (bottomTextPositionX, bottomTextPositionY)

  image = Image.fromarray(img)
  draw = ImageDraw.Draw(image)
  
  outlineRange = int(fontSize/15)
  for x in range(-outlineRange, outlineRange+1):
    for y in range(-outlineRange, outlineRange+1):
      draw.text((topTextPosition[0]+x, topTextPosition[1]+y), topString, (0,0,0), font=font)
      draw.text((bottomTextPosition[0]+x, bottomTextPosition[1]+y), bottomString, (0,0,0), font=font)
  draw.text(topTextPosition, topString, (255,255,255), font=font)
  draw.text(bottomTextPosition, bottomString, (255,255,255), font=font)

  return image

emb_dim = 100
attention_dim = 1024
hidden_dim = 1024
dropout = 0.5
encoder = Encoder()
decoder = Decoder(attention_dim=attention_dim,vocab_size=len(weights),embedding_dim=emb_dim,hidden_dim=hidden_dim)
decoder.load_glove(weights)


encoder_checkpoint = torch.load('model/project_encoder_good.pth', map_location='cpu')
decoder_checkpoint = torch.load('model/project_decoder_good.pth', map_location='cpu')
encoder.load_state_dict(encoder_checkpoint)
decoder.load_state_dict(decoder_checkpoint)


encoder.eval()
decoder.eval()


pun_dict = {'<PERIOD>':'.','<COMMA>':',','<QUOTATION_MARK>':'"','<SEMICOLON>':';','<EXCLAMATION_MARK>':'!','<QUESTION_MARK>':'?','<LEFT_PAREN>':'(','<RIGHT_PAREN>':')','<HYPHENS>':'--','<NEW_LINE>':'\n','<COLON>':':','<emp>':''}
args = {
   'output_dir': 'outputs/',
   'cache_dir': 'cache/',

   'fp16': False,
   'fp16_opt_level': 'O1',
   'max_seq_length': 128,
   'train_batch_size': 8,
   'eval_batch_size': 8,
   'gradient_accumulation_steps': 1,
   'num_train_epochs': 3,
   'weight_decay': 0,
   'learning_rate': 4e-5,
   'adam_epsilon': 1e-8,
   'warmup_ratio': 0.06,
   'warmup_steps': 0,
   'max_grad_norm': 1.0,
   'logging_steps': 50,
   'evaluate_during_training': False,
   'save_steps': 2000,
   'eval_all_checkpoints': True,
   'use_tensorboard': True,

   'overwrite_output_dir': True,
   'reprocess_input_data': True,
   "n_gpu": 0,
}

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
template_selection_module = ClassificationModel('bert', 'outputs/checkpoint-278000',num_labels=300,args = args,use_cuda=False)
 


def generate_meme(input):
  predictions, raw_outputs = template_selection_module.predict([input])

  candidates = raw_outputs[0].argsort()[::-1][:3]
  results = []
  result_texts = []
  for num in candidates:
    img_path = "images/" + templates_dict[temp_num2id[num]] +'.jpg'
    image = Image.open(img_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
      k = 3
      vocab_size = len(vocab_to_int)
  
      encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
      enc_image_size = encoder_out.size(1)
      encoder_dim = encoder_out.size(3)


      encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
      num_pixels = encoder_out.size(1)

      encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
      k_prev_words = torch.LongTensor([[vocab_to_int['<START>']]] * k)
      seqs = k_prev_words
  
      top_k_scores = torch.zeros(k, 1)
      seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size)

      complete_seqs = list()
      complete_seqs_alpha = list()
      complete_seqs_scores = list()

      step = 1
      h, c = decoder.init_hidden_state(encoder_out)

      while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
    
        awe, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores
    
        if step == 1:
          top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) 
        else:
          top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) 

        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size
    
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)


        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab_to_int['<END>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
          complete_seqs.extend(seqs[complete_inds].tolist())
          complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
          complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0:
          break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
          break
        step += 1

      i = complete_seqs_scores.index(max(complete_seqs_scores))
      seq = complete_seqs[i]
      alphas = complete_seqs_alpha[i]
      out = []
      for i in range(1,len(seq)-1):
        out.append(int_to_vocab[seq[i]])
    
    img = mpimg.imread(img_path)

    for i in range(len(out)):
      if out[i] in pun_dict.keys():
        out[i] = pun_dict[out[i]]  

    real_out = ' '.join(out)
    real_out = real_out.replace('<UNK>','')
    splited_out=real_out.split('\n')
    top = splited_out[0]
    bottom = splited_out[1]
    result_texts.append(real_out)
    results.append(make_meme(top, bottom, img))
    
  return results, result_texts

if __name__ == '__main__':
    input = "thank you, next"
    #input= "Work hard! or you'll be fired"
    generate_meme(input)