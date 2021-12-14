
$(document).ready(function() { 
	var xhr = null;

	jQuery.fn.extend({
		showError: function(message){
			$(this).append('<li class=\"alert alert-warning\">' + message + '</li>');
		},
		showInfo: function(message) {
			$(this).append('<li class=\"alert alert-light text-smaller\">' + message + '</li>');
		}
	});

	// Show the results on the screen.
	function showDemoResults(response){
        if(response != 0){
        	if(!("error" in response)) {
        		// Empty error messages if any and hide the progress indicator.
            	$('#loading-box').toggleClass("d-none");
				$('.flashes').empty()
				// Place the output image content.
				for (i = 1; i<4; i++) {
					$("#output-image-"+i).attr("src", "data:image/jpeg;base64," + response['output_image_'+i]);
					$('#output-text-'+i).html(response['output_text_'+i]);
				}
				
            	$('.flashes').showInfo(response['debug_str']);

            } else {
				// Show an error and hide progress indicator.
            	$('.flashes').showError(response['message']);
            	$('#loading-box').toggleClass("d-none");
            }

        }else{
        	// Show an error and hide progress indicator.
            $('.flashes').showError('There was an error on the server side');
            $('#loading-box').toggleClass("d-none");
        }
    }

    // This only happens when the server throws a serious error e.g. not a 200 HTTP error code.
    function showDemoError(response) {
		$('.flashes').showError('There was an error on the server side');
		$('#loading-box').toggleClass("d-none");
	}
	
	// Handle the text input trigger.
	$("#text-input-form").on("submit", function(e){
		// Prevent multiple AJAX calls. 
		if(xhr && xhr.readyState != 4){ xhr.abort(); e.preventDefault(); return false;}

		text_input = $("#text-input-1").val()
	  	// Clear any error messages and show loading bar.
		$('.flashes').empty()
  		$('#loading-box').toggleClass("d-none");
		// Prepare data to be sent with form.
		var form_data = new FormData();
		form_data.append('text_input', text_input);
		
		// Submit the form using an asynchronous call.
		xhr = $.ajax({
			url: 'simple-demo',
			type: 'post',
			data: form_data,
			contentType: false,
			processData: false,
			success: showDemoResults,
			error: showDemoError
		});

	  	// Avoid a browser POST request.
    	e.preventDefault();		
		return false;
	});


})