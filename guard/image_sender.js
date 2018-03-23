
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
    console.log('jquery loaded');
})();

function send_images_to_api(){

    // loop over every image
    $("img").each(function(idx) {

       // grab image source
       img_src = $("img")[idx].src;

       // send the image to api
       $.ajax({
            type: method,
            url: endpoint,
            data: { 'url': img_src },
            success: function(response)
            {
                // if successfull, append response as sibling
                success_callback(response);
                message = JSON.parse(response).message;
                $("<p>" + message + "</p>").insertAfter($("img")[idx]);
            },
            dataType: 'html'
        });
    });
}

function success_callback(responseObj){
    // Do something like read the response and show data
    data = JSON.parse(responseObj);
    alert('data = ' + data.data); // Only applicable to JSON response
}

(function() {
    send_images_to_api();
})();

