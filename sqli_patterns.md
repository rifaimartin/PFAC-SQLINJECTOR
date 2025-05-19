# Format file: pattern,weight
# Basic SQL injection patterns
' or,10
" or,10
' ||,10
" ||,10
= or,10
= ||,10
' =,10
' >=,10
' <=,10
' <>,10
" =,10
" !=,10
= =,10
= <,10
 >=,10
 <=,10
' union,10
' select,10
' from,10
union select,10
select from,10
' convert(,10
' avg(,10
' round(,10
' sum(,10
' max(,10
' min(,10
) convert(,10
) avg(,10
) round(,10
) sum(,10
) max(,10
) min(,10
' delete,10
' drop,10
' insert,10
' truncate,10
' update,10
' alter,10
, delete,10
; drop,80
; insert,15
; delete,15
, drop,80
; truncate,15
; exec,80
xp_cmdshell,100
; truncate,15
' ; update,15
like or,10
like ||,10
' %,10
like %,10
 %,10
</script>,10
</script >,10
union,10
select,10
drop,10
insert,10
delete,10
update,10
or 1=1,10
--,5
#,5
/*,5
*/,5
sleep(,15
benchmark(,15
count(*),10
information_schema.schemata,10
null,10
version(,15
current_user,15
outfile,80
load_file,80

# Critical patterns for DROP TABLE detection
drop table,80
; drop table,80
'; drop table,80
"; drop table,80
drop database,100
; drop database,100
'; drop database,100
"; drop database,100
truncate table,80
; truncate table,15
; execute,80
'; execute,80
; exec,80
'; exec,80

# More specific patterns for DROP TABLE users
drop table users,100
; drop table users,100
'; drop table users,100
"; drop table users,100
drop users,80
; drop users,80
'; drop users,80
"; drop users,80

# Special combined critical patterns with higher weights
'; drop table users;,100
; drop table users;,100
drop table users;,100
'; drop table users; --,100
; drop table users; --,100

# Additional SQL injection variations for critical cases
'; drop table users; -- /*,100
';drop table users;--,100
'; drop users;,100
;drop users,80