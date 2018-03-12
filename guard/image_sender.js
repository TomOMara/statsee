

// Invoke Jquery straight aways
(function() {
    // Load the script
    var script = document.createElement("SCRIPT");
    script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js';
    script.type = 'text/javascript';
    script.onload = function() {
        var $ = window.jQuery;
        // Use $ here...
    };
    document.getElementsByTagName("head")[0].appendChild(script);
})();


// Grab all images on page
// TODO: store the most local id of the image so we can dump api response back here
function img_find() {
    var imgs = document.getElementsByTagName("img");
    var imgSrcs = [];

    for (var i = 0; i < imgs.length; i++) {
        imgSrcs.push(imgs[i].src);
    }

    return imgSrcs;
}

// This will verify likeyhood of image being a line graph
function verify_images(){

}


function success(response){
    alert('success!');
    console.log(response);
}

function send_image_url_to_api(url){

    endpoint = 'http://127.0.0.1:5000/';

    method = 'POST';

    $.ajax({
        type: method,
        url: endpoint,
        data: url,
        success: success,
        dataType: 'html'
    });

}

image_url = 'hi.com'
send_image_url_to_api(image_url);
