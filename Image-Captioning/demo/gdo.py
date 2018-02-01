import os, subprocess, random, urllib, uuid, urlparse
from PIL import Image
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


"""
1. Install Flask: sudo pip install Flask
2. Curl:  curl http://127.0.0.1:5000/ to read
3. sudo apt-get install ufw
sudo ufw allow 5000

http://sudodev.cn/flask-external-access/

"""

# print(os.path.exists("run_inference_demo.py"))
# exit()

@app.route("/")
def main():
    os.putenv("CUDA_VISIBLE_DEVICES", "")
    return render_template('index.html')


@app.route("/getImages")
def get_image_path():
    # images = random.sample(os.listdir("/home/dsigpu4/Samba/im2txt/im2txt/data/mscoco/raw-data/val2014/"), 10)
    images = random.sample(os.listdir("/home/haodong/Workspace/image_captioning/data/mscoco/raw-data/val2014/"), 10)
    return jsonify(images)


@app.route("/getCaption")
def get_caption():
    # checkpoint_dir = "--checkpoint_path=/home/dsigpu4/Samba/im2txt/model/train"
    # vocab_file = "--vocab_file=/home/dsigpu4/Samba/im2txt/im2txt/data/mscoco/word_counts.txt"
    checkpoint_dir = "--checkpoint_path=/home/haodong/Workspace/image_captioning/model/train"
    vocab_file = "--vocab_file=/home/haodong/Workspace/image_captioning/data/mscoco/word_counts.txt"

    image_name = request.args.get('imageName')
    # image_file = "--input_files=/home/dsigpu4/Samba/im2txt/im2txt/data/mscoco/raw-data/val2014/%s" % (image_name)
    image_file = "--input_files=/home/haodong/Workspace/image_captioning/data/mscoco/raw-data/val2014/%s" % (image_name)
    param_str = "--checkpoint_path=%s   --vocab_file=%s  --input_files=%s" % (checkpoint_dir, vocab_file, image_file)

    # p = subprocess.Popen(["../bazel-bin/im2txt/run_inference", checkpoint_dir, vocab_file, image_file], stdout = subprocess.PIPE)
    p = subprocess.Popen(["../env2_2/bin/python", "/home/haodong/Workspace/image_captioning/run_inference_demo.py", checkpoint_dir, vocab_file, image_file], stdout = subprocess.PIPE)

    out = p.communicate()
    return out


@app.route("/getCaptionUrl")
def get_caption_url():
    image_url = request.args.get('imageUrl')
    image_name = str(uuid.uuid4())

    file_ext = os.path.splitext(os.path.basename(urlparse.urlsplit(image_url).path))[1]
    urllib.urlretrieve(image_url, "static/uploads/%s%s" % (image_name, file_ext))

    # if file_ext != ".jpg":
    im = Image.open("static/uploads/%s%s" % (image_name, file_ext))
    im.save("static/uploads/%s.jpg" % (image_name), "JPEG")

    # checkpoint_dir = "--checkpoint_path=/home/dsigpu4/Samba/im2txt/model/train"
    # vocab_file = "--vocab_file=/home/dsigpu4/Samba/im2txt/im2txt/data/mscoco/word_counts.txt"

    checkpoint_dir = "--checkpoint_path=/home/haodong/Workspace/image_captioning/model/train"
    vocab_file = "--vocab_file=/home/haodong/Workspace/image_captioning/data/mscoco/word_counts.txt"

    image_file = "--input_files=static/uploads/%s.jpg" % (image_name)
    param_str = "--checkpoint_path=%s   --vocab_file=%s  --input_files=%s" % (checkpoint_dir, vocab_file, image_file)

    # p = subprocess.Popen(["../bazel-bin/im2txt/run_inference", checkpoint_dir, vocab_file, image_file], stdout = subprocess.PIPE)
    p = subprocess.Popen(["../env2_2/bin/python", "/home/haodong/Workspace/image_captioning/run_inference_demo.py", checkpoint_dir, vocab_file, image_file], stdout = subprocess.PIPE)

    out = p.communicate()
    return out


if __name__ == "__main__":
    # app.run(port='5001')  # internal
    app.run(host='0.0.0.0', port='5002', threaded=True)    # external
