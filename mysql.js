let mysql = require('mysql');

const connection = mysql.createConnection({
    host: 'localhost',
    port: '3306',
    user: 'root',
    password: 'hwang261!',
    database: 'nutrition',
    debug: false
});

exports.returnFacts = function(name) {
    return new Promise(async function(resolve, reject){
        let sql = 'SELECT * from facts where name=?';
        let params = name;
        if(name.length>=2){
            for(let i=0; i<name.length-1;i++){
                sql += " or name=?";
            }
        }
        await connection.query(sql, params, function (err, rows, fields) {
            if (!err) {
                console.log("sql query success");
                console.log(rows);
                resolve(rows);
            }
            else{
                console.log('Error while performing Query.', err);
            }
        });
    });
}


