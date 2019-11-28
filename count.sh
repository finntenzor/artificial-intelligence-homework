find . -name "*.cu"|xargs cat|grep -v ^$|wc -l
find . -name "*.cpp"|xargs cat|grep -v ^$|wc -l
find . -name "*.h"|xargs cat|grep -v ^$|wc -l
