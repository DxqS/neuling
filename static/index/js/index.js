/**
 * Created by chk01 on 2017/7/3.
 */
$(function () {
    var time_left = parseInt($("#time_left").val());
    function countDown(btn) {
        function time() {
            if (time_left <= 0) {
                alert("BOOM!,您的免费学习时间已到。");
                window.location.href="/login"
            } else {
                btn.html('还剩' + time_left + '秒');
                time_left--;
                setTimeout(function () {
                    time()
                }, 1000)
            }
        }
        time()
    }
    countDown($("#left_time"))
});
