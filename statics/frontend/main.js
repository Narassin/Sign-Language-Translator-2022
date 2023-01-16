// =========================================================
// Project Sign Language Translator (Main Javascript)
// Version 1.3
// 10/1/2023
// Coded by: Ahmad Nuruddin Muksalmina (Narassin)
// =========================================================

//==================Translator Code=========================
// Define model and api path
let model;
const modelURL = 'http://localhost:5000/model';

// Variable initialization
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');
const prevbox = document.getElementById('preview-img')
const alpha = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y",'Z'," "]

// Predict Module
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

// Image Renderer
const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    reader.onload = () => {
        prevbox.innerHTML = `<div class="image-block">
                                    <img src="${reader.result}" class="image-block_loaded" id="source"/>
                              </div>`;
        preview.innerText = `${label}`;

    };
    reader.readAsDataURL(img);
};



// Event Listener 
fileInput.addEventListener("change",() => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " file", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => {preview.innerHTML = ""; prevbox.innerHTML = `<div class="image-block">
</div>`});

// ===========================MISCELLANEOUS==============================
// Darkmode
function darkmode() {
   var element = document.body;
   element.classList.toggle("dark-mode");
}

// Navigation bar
function openNav() {
    document.getElementById("NLinks").style.width = "250px";
    document.getElementById("shadow_opacity").style.display="block";
  }
  
  function closeNav() {
    document.getElementById("NLinks").style.width = "0";
    document.getElementById("shadow_opacity").style.display="none";
  }

// ==============================DICTIONARY====================================
//   Get the sign alphabet gallery and containers
      // Add an event listener to each sign image to show the corresponding container when clicked
function showA() {
    document.getElementById("Modal A").style.display = "flex";
};

function closeA() {
    document.getElementById("Modal A").style.display = "none";
};

function showB() {
    document.getElementById("Modal B").style.display = "flex";
};

function closeB() {
    document.getElementById("Modal B").style.display = "none";
};
function showC() {
    document.getElementById("Modal C").style.display = "flex";
};

function closeC() {
    document.getElementById("Modal C").style.display = "none";
};
function showD() {
    document.getElementById("Modal D").style.display = "flex";
};

function closeD() {
    document.getElementById("Modal D").style.display = "none";
};
function showE() {
    document.getElementById("Modal E").style.display = "flex";
};

function closeE() {
    document.getElementById("Modal E").style.display = "none";
};
function showF() {
    document.getElementById("Modal F").style.display = "flex";
};

function closeF() {
    document.getElementById("Modal F").style.display = "none";
};
function showG() {
    document.getElementById("Modal G").style.display = "flex";
};

function closeG() {
    document.getElementById("Modal G").style.display = "none";
};
function showH() {
    document.getElementById("Modal H").style.display = "flex";
};

function closeH() {
    document.getElementById("Modal H").style.display = "none";
};
function showI() {
    document.getElementById("Modal I").style.display = "flex";
};

function closeI() {
    document.getElementById("Modal I").style.display = "none";
};
function showJ() {
    document.getElementById("Modal J").style.display = "flex";
};

function closeJ() {
    document.getElementById("Modal J").style.display = "none";
};
function showK() {
    document.getElementById("Modal K").style.display = "flex";
};

function closeK() {
    document.getElementById("Modal K").style.display = "none";
};
function showL() {
    document.getElementById("Modal L").style.display = "flex";
};

function closeL() {
    document.getElementById("Modal L").style.display = "none";
};

function showM() {
    document.getElementById("Modal M").style.display = "flex";
};

function closeM() {
    document.getElementById("Modal M").style.display = "none";
};
function showN() {
    document.getElementById("Modal N").style.display = "flex";
};

function closeN() {
    document.getElementById("Modal N").style.display = "none";
};
function showO() {
    document.getElementById("Modal O").style.display = "flex";
};

function closeO() {
    document.getElementById("Modal O").style.display = "none";
};
function showP() {
    document.getElementById("Modal P").style.display = "flex";
};

function closeP() {
    document.getElementById("Modal P").style.display = "none";
};
function showQ() {
    document.getElementById("Modal Q").style.display = "flex";
};

function closeQ() {
    document.getElementById("Modal Q").style.display = "none";
};
function showR() {
    document.getElementById("Modal R").style.display = "flex";
};

function closeR() {
    document.getElementById("Modal R").style.display = "none";
};

function showS() {
    document.getElementById("Modal S").style.display = "flex";
};

function closeS() {
    document.getElementById("Modal S").style.display = "none";
};
function showT() {
    document.getElementById("Modal T").style.display = "flex";
};

function closeT() {
    document.getElementById("Modal T").style.display = "none";
};
function showU() {
    document.getElementById("Modal U").style.display = "flex";
};

function closeU() {
    document.getElementById("Modal U").style.display = "none";
};
function showV() {
    document.getElementById("Modal V").style.display = "flex";
};

function closeV() {
    document.getElementById("Modal V").style.display = "none";
};
function showW() {
    document.getElementById("Modal W").style.display = "flex";
};

function closeW() {
    document.getElementById("Modal W").style.display = "none";
};

function showX() {
    document.getElementById("Modal X").style.display = "flex";
}; 
 
function closeX() { 
    document.getElementById("Modal X").style.display = "none";
};
function showY() {
    document.getElementById("Modal Y").style.display = "flex";
};

function closeY() {
    document.getElementById("Modal Y").style.display = "none";
};
function showZ() {
    document.getElementById("Modal Z").style.display = "flex";
};

function closeZ() {
    document.getElementById("Modal Z").style.display = "none";
};
