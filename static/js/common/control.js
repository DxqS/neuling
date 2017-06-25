/**
 * Created by xumin on 16/6/28.
 */
var UI = {
    hanaro : {
        getById: function(id){
            return $('#'+id);
        },
        getByClass: function(className){
            return $('.'+className)
        }
    },
    registerId: function(id,even,fun,arr){
        if(UI.hanaro.getById(id)) UI.hanaro.getById(id).on(even,function(){fun(arr)});
    },
    registerIdNoObj: function(id,even,fun){
        if(UI.hanaro.getById(id)) UI.hanaro.getById(id).on(even,fun);
    },
    registerClass: function(className,even,fun,arr){
        if(UI.hanaro.getByClass(className)) UI.hanaro.getByClass(className).on(even,function(){fun(arr)});
    },
    registerClassNoObj: function(className,even,fun){
        if(UI.hanaro.getByClass(className)) UI.hanaro.getByClass(className).on(even,fun);
    },
};