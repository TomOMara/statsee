

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
function verified_images(){
    return img_find();
}


function success(response){
    alert('success!');
    console.log(response);
}

function send_verified_image_urls_to_api(){

    endpoint = 'http://127.0.0.1:5000/';
    method = 'POST';

    for (image_url in verified_images) {

        $.ajax({
            type: method,
            url: endpoint,
            data: image_url,
            success: success,
            dataType: 'html'
        });

    }
}


// We have an image located at https://imagebin.ca/v/3ueIn2A8o5JJ