let express = require('express');
let multer = require('multer');
let mysql = require('./mysql');
let fileNames = Array();
let {PythonShell} = require('python-shell');


function getPrediction() {
    return new Promise(   async function (resolve, reject) {
        let options = {
            mode: 'text',
            pythonPath:'',
            pythonOptions: ['-u'],
            scriptPath: '',
            args: fileNames
        };
        await PythonShell.run('./model/model.py', options, function(err, results){
            if(err) throw err;
            console.log(results)
            resolve(results);
        });


    });
}


let storage = multer.diskStorage({
    destination: function(req, file, cb){
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    }
});

let upload = multer({storage: storage});

let router = express.Router();

router.route('/').get(function (req, res) {
    res.render('index.html');
});

router.route('/').post(upload.array('photo',10),function(req,res){
    fileNames = Array();
    let files = req.files;

    for(let i=0;i<files.length;i++){
        console.dir(req.files[i]);
        fileNames.push(req.files[i].path);
    }

    id = req.body.id;
    let item = req.files[0].originalname

    getPrediction().then(function(predictions){
        return predictions

    }).then(function(name){
        mysql.returnFacts(name).then(function(result){
            res.json(result)
        })
    });


});



module.exports = router;