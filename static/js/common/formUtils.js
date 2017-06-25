(function ($) {
    /**
     * 表单扩展工具集
     */
    $.fn.formUtils = {
        formData: function (form) {
            var data = {};
            $(form).find('input,textarea,select').filter(':not([type="submit"],[type="reset"],[type="button"])').each(function () {
                if ($(this).attr('type') == 'radio') {
                    if ($(this).attr("checked") == 'checked') {
                        data[this.name] = $.trim($(this).val());
                    }
                } else if ($(this).attr('type') == 'checkbox') {
                    if ($(this).is(':checked')) {
                        if (typeof data[this.name] == 'undefined') {
                            data[this.name] = $.trim($(this).val());
                        } else {
                            data[this.name] = data[this.name].split('|');
                            data[this.name].push($.trim($(this).val()));
                            data[this.name] = data[this.name].join('|');
                        }
                    }
                } else {
                    data[this.name] = $.trim($(this).val());
                }
            });
            return data;
        },
        /**
         * Available validators
         */
        validators: {},
        addValidator: function (validator) {
            this.validators[validator.name] = {
                name: validator.name,
                fn: validator.fn,
                msg: validator.msg || '不正确',
                params: validator.params || {},
                acceptEmpty: ( typeof validator.acceptEmpty == 'undefined') ? true : validator.acceptEmpty
            };
        },
        empty: function (obj) {
            return typeof obj == 'undefined' || obj === null || obj === '';
        }
    };

    $.fn.validationSetup = function (setting) {
        var $form = $(this);
        var success = function (field, msg) {
            $(field).next().remove();
            $(field).parent().removeClass('has-error');
        };

        var error = function (field, msg) {
            var label, span, cgrp;
            if ($(field).parent().find('.label').html() == undefined) {
                $(field).parent().next().remove();
                cgrp = $(field).parent().parent();
                cgrp.removeClass('has-success').addClass('has-error');
                label = cgrp.find('label').html() + ' ';
                span = $('<span class="help-block col-sm-5">').append(label + msg);
                cgrp.append(span);
            } else {
                $(field).next().remove();
                cgrp = $(field).parent();
                cgrp.removeClass('has-success').addClass('has-error');
                label = cgrp.find('.label').html() + ' ';
                span = $('<span class="form-group form-warning">').append(label + msg);
            }
            cgrp.append(span);
        };

        var help = function (field, msg) {
            $(field).next().remove();
            var cgrp = $(field).parent();
            cgrp.removeClass('has-success').removeClass('has-error');
            var span = $('<span class="form-group form-warning">').append(msg);
            cgrp.append(span);
        };

        var config = $.extend({
            ruleAttribute: 'data-validation', // name of the attribute holding the validation rules
            errorMsgAttribute: 'data-validation-error-msg',
            helpMsgAttribute: 'data-validation-help',
            validateOnBlur: false,
            validateOnTextChange: true,
            showHelpOnFocus: true,
            successFn: success,
            errorFn: error,
            helpFn: help
        }, setting || {});

        if (config.validateOnBlur) {
            $form.find('[' + config.ruleAttribute + ']').blur(function () {
                $(this).validateInput(config);
            });
        }

        if (config.validateOnTextChange) {
            $form.find('[' + config.ruleAttribute + ']').change(function () {
                $(this).validateInput(config);
            });
        }

        if (config.showHelpOnFocus) {
            $form.find('[' + config.helpMsgAttribute + ']').focus(function () {
                config.helpFn(this, $(this).attr(config.helpMsgAttribute));
            });
        }

        $form.check = function () {
            var result = true;
            $form.find('input[' + config.ruleAttribute + '], textarea[' + config.ruleAttribute + ']').each(function () {
                if (!$(this).validateInput(config)) {
                    result = false;
                }
            });
            return result;
        };

        $form.reset = function () {
            $form.find('input[' + config.ruleAttribute + '], textarea[' + config.ruleAttribute + ']').each(function () {
                config.successFn($(this), null);
            });
        };

        return $form;
    };

    $.fn.validateInput = function (config) {
        var msg = ' ';
        var result = true;
        var errorMsg = $(this).attr(config.errorMsgAttribute);
        var validationRules = $(this).attr(config.ruleAttribute).split(" ");
        for (var i in validationRules) {
            var vdtor = $.fn.formUtils.validators[validationRules[i]];
            if (vdtor && typeof vdtor['fn'] == 'function') {
                var res = (vdtor['acceptEmpty'] && $(this).val() === '') || vdtor.fn(this, $(this).val(), vdtor['params']);
                if (!res) {
                    msg = msg + vdtor.msg + ' ';
                    result = false;
                }
            } else {
                console.warn('Using undefined validator "' + validationRules[i] + '"');
            }
        }

        msg = $.fn.formUtils.empty(errorMsg) ? msg : errorMsg;

        if (result) {
            config.successFn(this, msg);
        } else {
            config.errorFn(this, msg);
        }
        return result;
    };

    $.fn.formUtils.addValidator({
        name: 'require',
        fn: function (field, value, params) {
            return !$.fn.formUtils.empty(value) && !( typeof value.length !== 'undefined' && value.length === 0);
        },
        msg: '不能为空',
        acceptEmpty: false
    });

    $.fn.formUtils.addValidator({
        name: 'numeric',
        fn: function (field, value, params) {
            return /^\d*$/.test(value);
        },
        msg: '不能为非数字'
    });

    $.fn.formUtils.addValidator({
        name: 'email',
        fn: function (field, value, params) {
            return /^[a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(value);
        }
    });

    $.fn.formUtils.addValidator({
        name: 'phone',
        fn: function (field, value, params) {
            return /^0{0,1}(13|14|15|17|18)[0-9]{9}$/.test(value);
        },
        msg: '不是一个有效的手机号码'
    });

    $.fn.formUtils.addValidator({
        name: 'idcard',
        fn: function (field, value, params) {
            return $.fn.IdentityCodeValid(value);
        },
        msg: '不是一个有效的身份证号码'
    });

    $.fn.formUtils.addValidator({
        name: 'passport',
        fn: function (field, value, params) {
            return /^(1[45][0-9]{7}|[a-zA-Z][0-9]{8}|P[0-9]{7}|S[0-9]{7,8}|D[0-9]+)$/.test(value);
        },
        msg: '不是一个有效的护照号码'
    });

    $.fn.formUtils.addValidator({
        name: 'futureTime',
        fn: function (field, value, params) {
            return $.fn.CheckDate(value)
        },
        msg: '不能选择当日或之前的日期'
    });

    $.fn.CheckDate = function(date){
        var today = new Date();
        today.setHours(0);
        today.setMinutes(0);
        today.setSeconds(0);
        today.setMilliseconds(0);
        var deadline = today.getTime()/1000 + 3600*24;
        var choose_time = new Date(date).getTime()/1000;
        return choose_time >= deadline
    };


    $.fn.IdentityCodeValid = function(code) {
        var city = {
            11: "北京",
            12: "天津",
            13: "河北",
            14: "山西",
            15: "内蒙古",
            21: "辽宁",
            22: "吉林",
            23: "黑龙江 ",
            31: "上海",
            32: "江苏",
            33: "浙江",
            34: "安徽",
            35: "福建",
            36: "江西",
            37: "山东",
            41: "河南",
            42: "湖北 ",
            43: "湖南",
            44: "广东",
            45: "广西",
            46: "海南",
            50: "重庆",
            51: "四川",
            52: "贵州",
            53: "云南",
            54: "西藏 ",
            61: "陕西",
            62: "甘肃",
            63: "青海",
            64: "宁夏",
            65: "新疆",
            71: "台湾",
            81: "香港",
            82: "澳门",
            91: "国外 "
        };
        var tip = "";

        var pass = true;
         if (code == "") {
             tip = "请输入身份证号！";
             pass = false;
         }
        else if (!code || !/^\d{6}(18|19|20)?\d{2}(0[1-9]|1[012])(0[1-9]|[12]\d|3[01])\d{3}(\d|X|x)$/i.test(code)) {
            tip = "身份证号格式错误";
            pass = false;
        }

        else if (!city[code.substr(0, 2)]) {
            tip = "地址编码错误";
            pass = false;
        }
        else {
            //18位身份证需要验证最后一位校验位
            if (code.length == 18) {
                code = code.toUpperCase().split('');
                //∑(ai×Wi)(mod 11)
                //加权因子
                var factor = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
                //校验位
                var parity = [1, 0, 'X', 9, 8, 7, 6, 5, 4, 3, 2];
                var sum = 0;
                var ai = 0;
                var wi = 0;
                for (var i = 0; i < 17; i++) {
                    ai = code[i];
                    wi = factor[i];
                    sum += ai * wi;
                }
                var last = parity[sum % 11];
                if (parity[sum % 11] != code[17]) {
                    tip = "校验位错误";
                    pass = false;
                }
            } else {
                tip = "身份证号格式错误";
                pass = false;
            }
        }

        return pass;
    }

})(jQuery);

