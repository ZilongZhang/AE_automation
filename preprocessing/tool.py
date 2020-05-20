def PU_data_load(src_dir = './segmented',ratio = 3):
    pos_text = []
    neg_text = []
#-------------read from folder--------------------
    for root,dirs,files in os.walk(src_dir):
        for file in files:
            abs_path = os.path.join(root,file)
            with open(abs_path, 'rb') as f:
                    tmp_list = pickle.load(f)
            if 'pos' in file:
                pos_text.extend(tmp_list)
            elif 'neg' in file:
                neg_text.extend(tmp_list)
    neg_text = neg_text[:ratio * len(pos_text)]
#-----------generate labels for training data------------------
    pos_label = []
    for i in range(len(pos_text)):
        pos_label.append(1)
    neg_label = []
    for i in range(len(neg_text)):
        neg_label.append(0)
#------------combine pos and neg text together-----------
    neg_text.extend(pos_text)
    whole_text = neg_text
    neg_label.extend(pos_label)
    whole_label = neg_label
    return whole_text,whole_label