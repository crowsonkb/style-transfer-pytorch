"use strict";

let average;
let canLoadImage = true;
let done = false;
let ws;


function AverageTime(decay) {
    this.decay = decay;
    this.last_time = 0;
    this.value = 0;
    this.t = -1;

    this.get = () => {
        return this.value / (1 - Math.pow(this.decay, this.t));
    };

    this.update = (time) => {
        this.t += 1;
        if (this.t) {
            this.value *= this.decay;
            this.value += (1 - this.decay) * (time - this.last_time);
        }
        this.last_time = time;
        return this.get();
    };
}


function reloadImage() {
    if (canLoadImage) {
        canLoadImage = false;
        let width = $("#image").attr("width");
        let height = $("#image").attr("height");
        $("#image").attr("id", "backup-image");
        let img = $("<img id='image' style='display: none;'>");
        img.attr("src", "image?time=" + Date.now());
        img.attr("width", width);
        img.attr("height", height);
        img.on("load", onImageLoad);
        img.on("error", onImageError);
        $("#backup-image").after(img);
    }
}


function onImageLoad() {
    $("#backup-image").remove();
    $("#image").css("display", "");
    setTimeout(() => {canLoadImage = true;}, 100);
}


function onImageError() {
    $("#image").remove();
    $("#backup-image").css("display", "");
    ws.close();
}


function wsConnect() {
    let protocol = window.location.protocol.replace("http", "ws");
    ws = new WebSocket(protocol + "//" + window.location.host + "/websocket");

    ws.onopen = () => {
        $("#status").text("Waiting for the first iteration...");
    };

    ws.onclose = () => {
        if (!done) {
            $("#status").text("Lost the connection to the backend.");
            $("#status").css("display", "");
        }
    };

    ws.onerror = ws.onclose;

    ws.onmessage = (e) => {
        let msg = JSON.parse(e.data);
        let dpr = Math.min(window.devicePixelRatio, 2);
        switch (msg._type) {
        case "STIterate":
            $("#image").attr("width", msg.w / dpr);
            $("#image").attr("height", msg.h / dpr);
            $("#w").text(msg.w);
            $("#h").text(msg.h);
            $("#i").text(msg.i);
            $("#i-max").text(msg.i_max);
            $("#loss").text(msg.loss.toFixed(6));
            if (!average || msg.i == 1) {
                average = new AverageTime(0.9);
                $("#ips").text("0");
            }
            average.update(msg.time);
            if (average.t > 0) {
                $("#ips").text((1 / average.get()).toFixed(2));
            }
            if (msg.gpu_ram) {
                $("#gpu-ram").text((msg.gpu_ram / 1024 / 1024).toFixed());
                $("#gpu-wrap").css("display", "");
            }
            $("#status").css("display", "none");
            reloadImage();
            break;
        case "WIDone":
            $("#status").text("Iteration finished.");
            $("#status").css("display", "");
            done = true;
            $("#image").off();
            canLoadImage = true;
            reloadImage();
            ws.close();
            break;
        default:
            console.log(msg);
        }
    };
}

$(document).ready(() => {
    wsConnect();
});
