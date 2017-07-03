/**
 * Created by chk01 on 2017/7/3.
 */
$(function () {
    function show() {
        var a1 = parseInt(Math.random() * 9) + 1;
        var a2 = parseInt(Math.random() * 9) + 1;
        var sign = ["+", "-", "*"][parseInt(Math.random() * 3)];
        $("#uname").val(a1 + sign + a2 + "=?")
    }

    var iCount = setInterval(show, 1500);
    $('html').on("keydown", function (e) {
        if (e.keyCode == 13) {
            clearInterval(iCount);
            var pwd = $("#pwd").val();
            var uname = $("#uname").val().replace("=?", "");
//                var sg = uname.replace(/\d+/g,'');
//                var question = uname.split(sg);
//                alert(question[0][0]);
            if (parseInt(pwd) == eval(uname)) {
                $.post(window.location.pathname, function (res) {
                    if (res.status == 1) {
                        window.location.href = '/pic/index'
                    }
                }, 'json');
            } else {
                alert("太笨了！！！");
                $("#pwd").val("");
                iCount = setInterval(show, 1500)
            }
        }
    });
})
