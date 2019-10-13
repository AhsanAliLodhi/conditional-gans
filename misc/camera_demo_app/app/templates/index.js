function readURL(input,target) {
  if (input.files && input.files[0]) {
      var reader = new FileReader();

      reader.onload = function (e) {
          $(target)
              .attr('src', e.target.result)
              .attr('name',input.files[0].name)
              .width(256)
              .height(256);
      };

      reader.readAsDataURL(input.files[0]);
  }
}


function post(url, data, success) {
  var params = typeof data == 'string' ? data : Object.keys(data).map(
          function(k){ return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]) }
      ).join('&');

  var xhr = window.XMLHttpRequest ? new XMLHttpRequest() : new ActiveXObject("Microsoft.XMLHTTP");
  xhr.open('POST', url);
  xhr.onreadystatechange = function() {
      if (xhr.readyState>3 && xhr.status==200) { success(xhr.responseText); }
  };
  xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.send(params);
  return xhr;
}

function shift_input_image(){
  var image = document.getElementById("input_image").attributes.src;
  var shift_by = document.getElementById("shift_by").value;
  var src_abs_pos = document.getElementById("input_image_abs_pos").value;
  post("shift",{image:image.value,shift_by:shift_by},function(result){
    document.getElementById('output_image').src = result;
    var num = parseFloat(src_abs_pos) + parseFloat(shift_by)
    num = Math.round(num * 100) / 100
    document.getElementById('output_image_disp').innerText = num;
  });
}

function compare_camera(camera,abs_pos){
  image_input = document.getElementById("input_image").name;
  post("getcameraimage",{image:image_input,camera:camera},function(result){
    document.getElementById('compare_image').src = result;
    document.getElementById('compare_image_disp').innerText = abs_pos;
  });
}