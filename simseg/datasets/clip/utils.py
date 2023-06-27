import random

from io import BytesIO


def convert_img_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, "jpeg")
    imbytes = buffer.getvalue()
    return imbytes

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False

def process_caption(tokenizer, caption, train=True):
    tokens = tokenizer.tokenize(caption)
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                output_tokens.append(token)
                deleted_idx.append(len(output_tokens) - 1)
        else:
            # no masking token (will be ignored by loss function later)
            output_tokens.append(token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    restored_text = []
    for i in range(len(output_tokens)):
        if output_tokens[i] == '[MASK]':
            restored_text.append(output_tokens[i])
            continue
        if not is_subtoken(output_tokens[i]) and (i+1)<len(output_tokens) and is_subtoken(output_tokens[i+1]):
            restored_text.append(output_tokens[i] + output_tokens[i+1][2:])
            if (i+2)<len(output_tokens) and is_subtoken(output_tokens[i+2]):
                restored_text[-1] = restored_text[-1] + output_tokens[i+2][2:]
        elif not is_subtoken(output_tokens[i]):
            restored_text.append(output_tokens[i])

    return ' '.join(restored_text)
