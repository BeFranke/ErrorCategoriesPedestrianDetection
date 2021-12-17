/**
 * Source for Zoom-Logic and coordinate transforms and generally awesome:
 * http://phrogz.net/tmp/canvas_zoom_to_cursor.html
 */

const base_path = "cityscapes/leftImg8bit/val/"
function mod(n, m) {
    // modulo that does not give negative results
    return ((n % m) + m) % m;
}
const index_file = "eval/index.txt"

const COLOR_MAP = {
    "gt": {
        "env_occ": "red",
        "mxd_occ": "darkgreen",
        "crd_occ": "gold",
        "foreground": "hotpink",
        "other": "lime"
    },
    "dt": {
        "multi": "blue",
        "ghost": "dodgerblue",
        "scale": "aquamarine",
        "none": "linen"
    },
    "ignore": "black",
    "matchedGT": "silver"
}
const ID_MAP = [
    "env_occ",
    "mxd_occ",
    "crd_occ",
    "foreground",
    "other",
    "multi",
    "ghost",
    "scale",
    "none",
    "ignore",
    "silver"
]

function draw_bounding_box(ctx, xmin, ymin, w, h, color) {
    // console.log(`draw [${xmin}, ${ymin}, ${w}, ${h}]!`);
    ctx.beginPath();
    ctx.lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.rect(xmin, ymin, w, h);
    ctx.stroke();
}

// (entry) => entry.replace(" IoU", "").split(" @ ").join("___")
function filename_from_dropdown(entry) {
    let entry_split = entry.split(" @ ");
    let model_name = entry_split[0];
    let iou = entry_split[1].split(" IoU")[0];
    let setup = entry_split[1].split("(")[1].split(")")[0];
    return model_name + "___" + iou + "___" + setup
}

class ErrorVisualizer {
    constructor(canvas, boxCanvas) {
        this.json = null;
        this.canvas = canvas;
        this.imgCtx = canvas.getContext("2d");
        this.canvas.width = window.innerWidth * 0.8;
        this.canvas.height = window.innerHeight * 0.8;
        this.boxCanvas = boxCanvas;
        this.boxCtx= boxCanvas.getContext("2d");
        this.boxCanvas.width = this.canvas.width;
        this.boxCanvas.height = this.canvas.height;
        this.curImgId = 1;
        this.n_imgs = null;
        this.curImg = new Image();
        this.lblCaption = document.getElementById("caption");
        this.lblXY = document.getElementById("xy");
        this.curConf = null;
        this.lastX = this.canvas.width / 2;
        this.lastY = this.canvas.height / 2;
        this.dragStart = null;
        // this.dragged = null;
        this.scaleFactor = 1.1;
        this.curPath = "";
        this.show = new Array(10).fill(true);
    }
    run(json) {
        this.json = json;
        this.n_imgs = Object.keys(json.imgs).length;
        let path = this.getImagePathFromId(this.curImgId);
        this.redraw(path, 0.5);
    }
    getImagePathFromId(id) {
        return base_path + this.json.imgs[id].im_name.split("_")[0] + "/" + this.json.imgs[id].im_name + ".png"
    }
    _drawBoundingBoxes(confThres) {
        const gts = this.json.gts[this.curImgId];
        const dts = this.json.dts[this.curImgId];
        let relevant_dts, relevant_gts;
        if(!this.show[9]) {
            relevant_dts = dts.filter(d => d["score"] > confThres && !d["matched_gt_ignore"]);
            let relevant_dt_ids = relevant_dts.map(d => d["id"]);
            relevant_gts = gts.filter(g => !relevant_dt_ids.includes(g["matched_dt"]) || g["ignore"]);
        } else {
            relevant_dts = dts.filter(d => d["score"] > confThres);
            relevant_gts = gts;
        }
        relevant_gts.forEach(x => this.drawBoundingBox(x, this.boxCtx, "gt", x["ignore"]))
        relevant_dts.forEach(x => this.drawBoundingBox(x, this.boxCtx, "dt"))
    }
    _draw(confThres) {
        this.imgCtx.drawImage(this.curImg, 0, 0,
            this.canvas.width, this.canvas.height);
        this.redrawBoxes(confThres);
    }
    redraw(path, confThres) {
        const p1 = this.imgCtx.transformedPoint(0, 0);
        const p2 = this.imgCtx.transformedPoint(this.canvas.width, this.canvas.height);
        this.imgCtx.clearRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        if(this.curPath !== path) {
            // only reload file if it has really changed
            this.curPath = path;
            this.curImg.src = path;
            this.curImg.onload = () => {
                this._draw(confThres)
            }
            let parts = path.split("/")
            this.lblCaption.innerHTML = parts[parts.length - 1].split(".")[0]
        }
        else {
            this._draw(confThres);
        }

    }
    redrawBoxes(confThres) {
        const p1 = this.boxCtx.transformedPoint(0, 0);
        const p2 = this.boxCtx.transformedPoint(this.boxCanvas.width, this.boxCanvas.height);
        this.boxCtx.clearRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        this.curConf = confThres;
        this._drawBoundingBoxes(confThres);
    }
    nextId() {
        let _id = this.curImgId - 1;
        _id = mod(_id + 1, this.n_imgs);
        _id = _id + 1;
        return _id;
    }
    previousId() {
        let _id = this.curImgId - 1;
        _id = mod(_id - 1, this.n_imgs);
        _id = _id + 1;
        return _id
    }
    drawBoundingBox(boxInfo, ctx, cmapKey, ignore = false) {
        // the !original! image scale
        const imgWidth = this.json.imgs[this.curImgId]["width"];
        const imgHeight = this.json.imgs[this.curImgId]["height"];
        let xmin = boxInfo["bbox"][0], ymin = boxInfo["bbox"][1], w = boxInfo["bbox"][2], h = boxInfo["bbox"][3]
        // make BB relative
        xmin /= imgWidth;
        ymin /= imgHeight;
        w /= imgWidth;
        h /= imgHeight;
        // compute coordinates in canvas coordinates
        xmin *= this.boxCanvas.width;
        ymin *= this.boxCanvas.height;
        w *= this.boxCanvas.width;
        h *= this.boxCanvas.height;

        let show = this.show[ID_MAP.findIndex(e => e === boxInfo["error_type"])] && !ignore || this.show[9] && ignore;
        if(show) {
            if (!ignore) {
                draw_bounding_box(this.boxCtx, xmin, ymin, w, h, COLOR_MAP[cmapKey][boxInfo["error_type"]]);
            } else {
                draw_bounding_box(this.boxCtx, xmin, ymin, w, h, COLOR_MAP["ignore"]);
            }
        }

    }
    changeModel(json) {
        this.json = json;
        this.redrawBoxes(this.curConf);
    }
    _zoom(clicks) {
        let pt_i = this.imgCtx.transformedPoint(this.lastX, this.lastY);
        let pt_b = this.boxCtx.transformedPoint(this.lastX, this.lastY);
        this.imgCtx.translate(pt_i.x, pt_i.y);
        this.boxCtx.translate(pt_b.x, pt_b.y);
        let factor = Math.pow(this.scaleFactor, clicks);
        this.imgCtx.scale(factor, factor);
        this.boxCtx.scale(factor, factor);
        this.canvScale *= factor;
        this.imgCtx.translate(-pt_i.x, -pt_i.y);
        this.boxCtx.translate(-pt_b.x, -pt_b.y);
        this.redraw(this.curPath, this.curConf);
    }

    _handleMouseDown(evt) {
        document.body.style.mozUserSelect = document.body.style.userSelect = 'none';
        this.lastX = evt.offsetX || (evt.pageX - this.canvas.offsetLeft);
        this.lastY = evt.offsetY || (evt.pageY - this.canvas.offsetTop);
        this.dragStart = this.imgCtx.transformedPoint(this.lastX, this.lastY);
    }
    _handleMouseUp() {
        this.dragStart = null;
    }
    _handleMouseMove(evt) {
        this.lastX = evt.offsetX || (evt.pageX - this.canvas.offsetLeft);
        this.lastY = evt.offsetY || (evt.pageY - this.canvas.offsetTop);
        let pt_i = this.imgCtx.transformedPoint(this.lastX, this.lastY);
        let pt_b = this.boxCtx.transformedPoint(this.lastX, this.lastY);
        let imgX = Math.round(pt_i.x / this.canvas.width * this.json.imgs["1"].width)
        let imgY = Math.round(pt_i.y / this.canvas.height * this.json.imgs["1"].height)
        this.lblXY.innerHTML = `X=${String("      " + imgX).slice(-5)}, 
                                Y=${String("      " + imgY).slice(-5)}`;
        if (this.dragStart) {
            this.imgCtx.translate(pt_i.x - this.dragStart.x, pt_i.y - this.dragStart.y);
            this.boxCtx.translate(pt_b.x - this.dragStart.x, pt_b.y - this.dragStart.y);
            this.redraw(this.curPath, this.curConf);
        }
    }
    _handleMouseWheel(evt) {
        let delta = evt.wheelDelta ? evt.wheelDelta / 40 : evt.detail ? -evt.detail : 0;
        if(delta)
            this._zoom(delta);
    }
    _handleLegend(idx) {
        this.show[idx] = !this.show[idx];
        this.redrawBoxes(this.curConf);
    }
}

window.onload = function() {
    // get index file
    fetch(index_file)
        .then(response => response.text())
        .then(text => main(text.split('\n')));
}


function main(index) {
    // create ErrorVisualizer to hold state
    let ev = new ErrorVisualizer(
        document.getElementById("imageCanvas"),
        document.getElementById("boxCanvas")
    );
    // load the first json-file
    let s = index[0];
    fetch("eval/" + s)
      .then(response => response.json())
      .then(json => {
          ev.run(json)
        // populate data list
        let datalist = document.getElementById("imageList");
        for(let key in ev.json.imgs) {
            let option = document.createElement('option');
            option.value = ev.json.imgs[key].im_name;
            datalist.appendChild(option);
        }
      });
    // slider logic
    let slider = document.getElementById("confSlider");
    let confLbl = document.getElementById("confLbl");
    confLbl.innerHTML = `Confidence Threshold: ${slider.value / 100}`;
    slider.oninput = function() {
        ev.redrawBoxes(this.value / 100);
        confLbl.innerHTML = `Confidence Threshold: ${this.value / 100}`;
    }
    slider.value = 50

    // button logic
    let nextButton = document.getElementById("btnForward");
    let prevButton = document.getElementById("btnBackward");
    let goButton = document.getElementById("go");
    // let toggleButton = document.getElementById("tglIgnore");
    nextButton.onclick = function () {
        let id = ev.nextId();
        ev.curImgId = id;
        let path = ev.getImagePathFromId(id);
        ev.redraw(path, ev.curConf);
    }
    prevButton.onclick = function () {
        let id = ev.previousId();
        ev.curImgId = id;
        let path = ev.getImagePathFromId(id);
        ev.redraw(path, ev.curConf);
    }
    goButton.onclick = function () {
        let toField = document.getElementById("goTo");
        if(toField.value) {
            let id = "" + Object.keys(ev.json.imgs).find(key => ev.json.imgs[key].im_name === toField.value);
            ev.curImgId = id;
            let path = ev.getImagePathFromId(id);
            ev.redraw(path, ev.curConf);
        }
    }

    // interactive legend
    let legend = [
        ["lgndEnv", "red"], ["lgndMxd", "darkgreen"], ["lgndCrd", "gold"], ["lgndFgd", "hotpink"],
        ["lgndStd", "lime"], ["lgndMtd", "blue"], ["lgndGst", "dodgerblue"],
        ["lgndScl", "aquamarine"], ["lgndMch", "linen"], ["lgndIgn", "black"]
    ];
    for(let i = 0; i < legend.length; ++i) {
        let e = legend[i];
        let code = e[0];
        let color = e[1];
        let btn = document.getElementById(code);
        btn._pressed = false;
        btn.style.backgroundColor = color;
        btn.onclick = function () {
            ev._handleLegend(i);
            if(!this._pressed) this.style.backgroundColor = "gray";
            else this.style.backgroundColor = color;
            this._pressed = !this._pressed;
        }
    }

    // dropdown logic
    let model_chooser = document.getElementById("model-selector");
    index.forEach(m => {
        let parts = m.split("___");
        let text = `${parts[0]} @ ${parts[1]} IoU (${parts[2]})`;
        fetch("eval/" + filename_from_dropdown(text))
            .then(response => {
                if(response.ok) {
                    let option = document.createElement("option");
                    option.text = text;
                    model_chooser.add(option);
                    if(m === index[0]) {
                        model_chooser.value = text;
                    }
                }
            });
    });
    model_chooser.addEventListener(
        'change',
        function() {
            console.log(this.value);
            fetch("eval/" + filename_from_dropdown(this.value))
                .then(response => response.json())
                .then(json => ev.changeModel(json));
        }
    );
    ev.canvas.addEventListener("mousedown", function(e) {ev._handleMouseDown(e)}, false);
    ev.canvas.addEventListener("mousemove", function(e) {ev._handleMouseMove(e)}, false);
    ev.canvas.addEventListener("mouseup", function(e) {ev._handleMouseUp()}, false);
    ev.canvas.addEventListener("mousewheel", function(e) {ev._handleMouseWheel(e)}, false);
    ev.canvas.addEventListener("DOMMouseScroll", function(e) {ev._handleMouseWheel(e)}, false);
    ev.boxCanvas.addEventListener("mousedown", function(e) {ev._handleMouseDown(e)}, false);
    ev.boxCanvas.addEventListener("mousemove", function(e) {ev._handleMouseMove(e)}, false);
    ev.boxCanvas.addEventListener("mouseup", function(e) {ev._handleMouseUp()}, false);
    ev.boxCanvas.addEventListener("mousewheel", function(e) {ev._handleMouseWheel(e)}, false);
    ev.boxCanvas.addEventListener("DOMMouseScroll", function(e) {ev._handleMouseWheel(e)}, false);

    trackTransforms(ev.imgCtx);
    trackTransforms(ev.boxCtx);

    // Adds ctx.getTransform() - returns an SVGMatrix
	// Adds ctx.transformedPoint(x,y) - returns an SVGPoint
	function trackTransforms(ctx){
        let svg = document.createElementNS("http://www.w3.org/2000/svg", 'svg');
        let xform = svg.createSVGMatrix();
        ctx.getTransform = function(){ return xform; };

        let savedTransforms = [];
        let save = ctx.save;
        ctx.save = function(){
			savedTransforms.push(xform.translate(0,0));
			return save.call(ctx);
		};
        let restore = ctx.restore;
        ctx.restore = function(){
			xform = savedTransforms.pop();
			return restore.call(ctx);
		};

        let scale = ctx.scale;
        ctx.scale = function(sx,sy){
			xform = xform.scaleNonUniform(sx,sy);
			return scale.call(ctx,sx,sy);
		};
        let rotate = ctx.rotate;
        ctx.rotate = function(radians){
			xform = xform.rotate(radians*180/Math.PI);
			return rotate.call(ctx,radians);
		};
        let translate = ctx.translate;
        ctx.translate = function(dx,dy){
			xform = xform.translate(dx,dy);
			return translate.call(ctx,dx,dy);
		};
        let transform = ctx.transform;
        ctx.transform = function(a,b,c,d,e,f){
            const m2 = svg.createSVGMatrix();
            m2.a=a; m2.b=b; m2.c=c; m2.d=d; m2.e=e; m2.f=f;
			xform = xform.multiply(m2);
			return transform.call(ctx,a,b,c,d,e,f);
		};
        let setTransform = ctx.setTransform;
        ctx.setTransform = function(a,b,c,d,e,f){
			xform.a = a;
			xform.b = b;
			xform.c = c;
			xform.d = d;
			xform.e = e;
			xform.f = f;
			return setTransform.call(ctx,a,b,c,d,e,f);
		};
        let pt = svg.createSVGPoint();
        ctx.transformedPoint = function(x,y){
			pt.x=x; pt.y=y;
			return pt.matrixTransform(xform.inverse());
		}
	}
}
