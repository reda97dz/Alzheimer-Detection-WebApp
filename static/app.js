// UI elements
const uploadBtn = document.querySelector('#upload');
const sumbitBtn = document.querySelector('#submitBtn');
const uploadTxt = document.querySelector('#uploadImg');
const positive = document.querySelector('#positive');
const negative = document.querySelector('#negative');
const prgPos = document.querySelector('#prgPos');
const prgNeg = document.querySelector('#prgNeg');
const predict = document.querySelector('#tryAgain');
const loading = document.querySelector('#loading');

predict.addEventListener('mousedown', function(e) {
    if(e.target.id==='predictAgain'){
        window.location.reload();
    }
});


function showInputImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        uploadBtn.style.display = 'none';
        uploadTxt.style.display = 'none';
        $('#imageInputDiv').className += 'text-center'
        reader.onload = function (e) {
            $('#inputImg')
                .attr('src', e.target.result);
        };
        // uploadTxt.textContent = 'Input image';
        reader.readAsDataURL(input.files[0]);
    }
}

$("form").submit(function(evt){
    evt.preventDefault();
    sumbitBtn.style.display = 'none';
    loading.style.display = 'block';
    var formData = new FormData($(this)[0])
    $.ajax({
        url: '/predict/',
        type: 'POST',
        data: formData,
        async: true,
        cache: false,
        contentType: false,
        enctype: 'multipart/form-data',
        processData: false,
        success: function (response){
            loading.style.display = 'none';
            var pos = parseFloat(response);
            var neg = 1-pos;
            negative.innerHTML = neg.toFixed(2);
            positive.innerHTML = pos.toFixed(2);
            prgPos.style.width = pos.toFixed(2)*100 + '%'
            prgNeg.style.width = neg.toFixed(2)*100 + '%'
            predictAgain.style.display = 'block';
        }
    });
});