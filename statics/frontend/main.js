let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');
const alpha = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y",'Z'," "]

const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
    const files = fileInput.files;
    console.log(files);
    

    [...files].map(async (img) => {
        const data = new FormData();
        data.append('file', img);

        const processedImage = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => {
                return tf.tensor2d(result['image'],[24,1],"float32");
            });

        // shape has to be the same as it was for training of the model
        console.log(typeof(processedImage));
        const prediction = model.predict(processedImage.reshape([ 1 ,24, 1]));
        const label = prediction.argMax(axis = 1).dataSync()[0];
        const tag = alpha[label-1]
        renderImageLabel(img, tag);
        
    })
};

const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    reader.onload = () => {
        preview.innerHTML += `<div class="image-block">
                                      <img src="${reader.result}" class="image-block_loaded" id="source"/>
                                       <h2 class="image-block__label">${label}</h2>
                              </div>`;

    };
    reader.readAsDataURL(img);
};


fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");
