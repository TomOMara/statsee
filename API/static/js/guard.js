/*
 * This is a JavaScript Scratchpad.
 *
 * Enter some JavaScript, then Right Click or choose from the Execute Menu:
 * 1. Run to evaluate the selected text (Cmd-R),
 * 2. Inspect to bring up an Object Inspector on the result (Cmd-I), or,
 * 3. Display to insert the result in a comment after the selection. (Cmd-L)
 */

// Invoke Jquery straight aways
(function() {
    // Load the script
    var bootstrap_script = document.createElement("LINK");
    bootstrap_script.href = 'https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.0.0/css/bootstrap.min.css';
    bootstrap_script.rel = "stylesheet";
    document.getElementsByTagName("head")[0].appendChild(bootstrap_script);

    var script = document.createElement("SCRIPT");
    script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js';
    script.type = "text/javascript";
    script.onload = function() {
        var $ = window.jQuery;
        // Use $ here...
        send_images_to_api();
    };
    document.getElementsByTagName("head")[0].appendChild(script);
    alert('jquery loaded');


})();

function send_images_to_api(){
    endpoint = 'http://127.0.0.1:5000/';
    method = 'POST';

    
    $("img").each(function(idx) {
       let img_src = $("img")[idx].src;
       console.log('looking at ', img_src);
       let current_index = idx;
       $.ajax({
           type: method,
           url: endpoint,
           data: { 'url': img_src },
           success: function(response)
           {
                message = JSON.parse(response).message;
                console.log(img_src + message);
                data = JSON.parse(response).data;
                if (data) {
                     $("<div class=\"alert alert-success\" role=\"alert\" style='margin: auto'>\n<p>" + data + "</p></div>").insertAfter($("img")[idx]);
                }
           },
           failure: function(f_response) {
               console.log('failed :(');
           },
           dataType: 'html'
        });
    });
}