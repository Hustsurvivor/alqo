SELECT
    COUNT(*)
FROM
    votes AS v,
    posts AS p,
    comments AS c,
    posthistory AS ph
Where
    c.postid = p.id
    AND p.id = ph.postid
    AND p.id = v.postid
    AND p.FavoriteCount != 22
    AND p.AnswerCount > 3
    AND ph.PostHistoryTypeId < 16
    AND v.CreationDate != 1371376921
    AND c.Score < 87
    AND c.CreationDate != 1408027181;