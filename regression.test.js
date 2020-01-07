const regression = require("./regression")

test('num_pts = 10 createData return 10 values',()=>{
    expect(app.createData(10).length).toBe(10)
})
