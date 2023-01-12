
// -----------------------------------------------------database

//
//

use HW_info;

db;

document=({student: 'Yidow Feng',
    class: 'CS777',
    professor: 'Farshid Alizadeh-Shabdiz',
    hw_name: 'termpaper',
    tags: ['mongodb', 'database', 'NoSQL'],
    points: 100
})
db.mycol.insertOne(document)

do2 = ({
   _id: 122,
   name: "Toy Story (1995)",
   info: {
              rating: "4.5",
              tage: "excellent",
              genres: "Adventure|Animation|Children|Comedy|Fantasy"
            }
})
db.movies.insertOne(do2)

movie_1 = ({
   movie_id: 122,
   name: "Toy Story (1995)"
})

rating_1 = ({
   user_id: 1,
   points: 4.5,
   timestamp: 1136418616
})

rating_2 = ({
   user_id: 2,
   points: 5,
   timestamp: 844416936
})

movie_1 = ({
   movie_id: 122,
   name: "Toy Story (1995)",
   ratings:[
       {
       user_id: 1,
       points: 4.5,
       timestamp: 1136418616
    },
    {
       user_id: 2,
       points: 5,
       timestamp: 844416936
    }
   ]
})

movie_1 = ({
   movie_id: 8,
   name: "Tom and Huck (1995)",
   ratings:{
       genre: Adventure,
       points: 5,
    }
    })

movie_2 = ({
   movie_id: 2,
   name: "Jumanji (1995)",
   ratings:{
       genre: Adventure,
       points: 5,
    }
    })

advanture = ({
       genre: Adventure,
       points: 5,
       movies:[2, 8]
})