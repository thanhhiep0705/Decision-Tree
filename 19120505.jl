using CSV
using DataFrames
using Random

#Đọc dữ liệu từ file csv trả về dataframe
function load_data(filename)
    data = CSV.read(filename, DataFrame)
    data = data[:,2:ncol(data)]
    return data
end

#Chia data ban đầu thành tập train và tập test
function split_data(data)
    data_shuffled = data[shuffle(1:nrow(data)), :]
    train_data = data_shuffled[1:Int64(nrow(data) * 2 / 3), :]
    test_data = data_shuffled[Int64(nrow(data) * 2 / 3)+1:nrow(data_shuffled), :]
    return train_data, test_data
end

#Tìm ra tất cả các giá trị của 1 thuộc tính trong dữ liệu
function get_label(data, attribute)
    uniq = []
    for i in data[!,attribute]
        if i in uniq
        else
            push!(uniq, i)
        end
    end
    return sort(uniq)
end

#Tính entropy của tập dữ liệu
function entropy(data)
    label = get_label(data, ncol(data) )
    values = 0
    for i in label
        temp = nrow(data[data[!,ncol(data)] .== i, :] )
        values = values + (temp/nrow(data))*log2(temp/nrow(data))
    end
    return -values
end

#Tính information gain của 1 thuộc tính trong tập dữ liệu dựa trên cutoff được truyền vào
function info_gain(data, attribute, cutoff)
    left = data[data[!,attribute] .<= cutoff[attribute], :]
    right = data[data[!,attribute] .> cutoff[attribute], :]
    gain = entropy(data) -  (nrow(left)/nrow(data))*entropy(left) - (nrow(right)/nrow(data))*entropy(right)
    return gain
end

#Tách tập dữ liệu ban đầu dựa vào thuộc tính và cutoff 
#Tất cả các mẫu có giá trị của attribute > cutoff thành 1 tập, ngược lại là 1 tập.
#Xóa đi cột thuộc tính đó
function split_example(data, attribute, cutoff)
    left = data[data[!,attribute] .<= cutoff[attribute], :]
    right = data[data[!,attribute] .> cutoff[attribute], :]
    left = select!(left, Not(attribute))
    right = select!(right, Not(attribute))
    return left, right
end

#Tìm ra thuộc tính tốt nhất(info_gain lớn nhất)
function find_maxfit(data, cutoff)
    if entropy(data) == 0
        return data[!,ncol(data)][1], -1000
    end
    max = 1
    for i in 1:ncol(data) - 1
        gain = info_gain(data, i, cutoff)
        if gain > info_gain(data, max, cutoff)
            max = i
        end
    end
    return names(data)[max], cutoff[max]
end

#Tìm ngưỡng cutoff sao cho entropy là nhỏ nhất
function find_cutoff(data)
    result = []
    for i in 1:ncol(data) -1
        label = get_label(data, i)
        max = 0
        temp = 0
        for j in label
            left =  data[data[!,i] .<= j, :]
            right = data[data[!,i] .> j, :]
            gain = entropy(data) -  (nrow(left)/nrow(data))*entropy(left) - (nrow(right)/nrow(data))*entropy(right)
            if gain > max
                max = gain
                temp = j
            end
        end
        push!(result, temp)
    end
    return result
end

#Tìm vị trí của 1 phần tử trong mảng
function find_index(list, values)
    for i in 1:length(list)
        if list[i] == values
            return i
        end
    end
end

#Cấu trúc dữ liệu để lưu cây
mutable struct Node
    name::String #Tên thuộc tính
    cutoff::Float64 #Ngưỡng cutoff của thuộc tính đó
    leftnode #Node con bên trái của node hiện tại(giá trị nhỏ hơn cutoff)
    rightnode #Node con bên phải của node hiện tại(giá trị lớn hơn cutoff)
    leafnode::Bool # Đánh dấu là Node lá hay không
    answer::String #Nếu là Node lá thì trả về kết quả
end

#Thay đổi thông tin của 1 Node
function set_info(p::Node, name, cutoff)
    p.name = name
    p.cutoff = cutoff
end

#Hàm chính để xây dựng cây
function fit(data, cutoff)
    Root = Node("",0,nothing,nothing,false,"")
    name, values = find_maxfit(data, cutoff)
    set_info(Root,name, values)
    loop(data, cutoff, Root)
    return Root
end

#Hàm gọi đệ quy để tìm tất cả các con của 1 Node
function loop(data, cutoff, node::Node)
    if node.leafnode == 1
        return
    end
    attribute = find_index(names(data), node.name)
    left, right = split_example(data, attribute,cutoff)
    if ncol(left) == 1 # Xử lí nếu tập chỉ còn 1 thuộc tính
        left_answer = get_label(left, 1)
        right_answer = get_label(right,1)
        left_max = -1
        left_best = ""
        right_max = -1
        right_best = ""
        for i in left_answer
            temp =  nrow(left[left[!,1] .== i, :])
            if temp > left_max
                left_max = temp
                left_best = i
            end
        end
        for i in right_answer
            temp =  nrow(left[left[!,1] .== i, :])
            if temp > right_max
                right_max = temp
                right_best = i
            end
        end
        a = Node("",0, nothing,nothing,true,left_best)
        b = Node("", 0, nothing,nothing,true,right_best)
        node.leftnode = a
        node.rightnode = b
        return
    end
    left_cutoff = find_cutoff(left)
    right_cutoff = find_cutoff(right)
    leftchild_name, leftchild_cutoff = find_maxfit(left, left_cutoff)
    rightchild_name, rightchild_cutoff = find_maxfit(right,right_cutoff )
    if leftchild_cutoff == -1000 
        a = Node("",0, nothing, nothing, true,leftchild_name)
    else
        a = Node(leftchild_name, leftchild_cutoff, nothing, nothing, false,"")
    end
    if rightchild_cutoff == -1000
        b = Node("",0, nothing, nothing, true,rightchild_name)
    else
        b = Node(rightchild_name, rightchild_cutoff, nothing, nothing, false,"")
    end
    node.leftnode = a
    node.rightnode = b
    loop(left, left_cutoff, node.leftnode)
    loop(right, right_cutoff, node.rightnode)
end

#Hàm vẽ cây quyết định
function visualize(p::Node, level)
    level += 1
    if p.leafnode == 1
        for i in 1:level
            print("     ")
        end
        println("-> " ,p.answer)
        return
    else
        for i in 1:level
            print("     ")
        end
        println(p.name , " > ", p.cutoff)
        visualize(p.leftnode, level)   
        for i in 1:level
            print("     ")
        end
        println(p.name , " <= ", p.cutoff)
        visualize(p.rightnode, level)    
    end
end

#Dự đoán kết quả của 1 mẫu dựa trên cây quyết định
function predict_1sample(node::Node, x_test)
    if node.leafnode == 1
        return node.answer
    end
    if x_test[node.name] > node.cutoff
        return predict_1sample(node.rightnode,x_test)
    else
        return predict_1sample(node.leftnode, x_test)
    end
end

#Dự đoán kết quả của tất cả các mẫu trong tập test 
function predict(Root::Node, test_data)
    y_hat = []
    for i in 1:nrow(test_data)
        x_test = test_data[i,:]
        push!(y_hat, predict_1sample(Root, x_test))
    end
    return y_hat
end

#Hàm tính độ chính xác dựa trên độ đo accuracy
function accuracy(test_data, y_hat)
    count = 0
    for i in 1:nrow(test_data)
        if y_hat[i] == test_data[i,:][ncol(test_data)]
            count += 1
        end
    end
    return count/nrow(test_data)
end


#Thực hiện chương trình bằng cách gọi các hàm chức năng trên
data = load_data("Iris.csv")
train_data, test_data = split_data(data)
cutoff = find_cutoff(train_data)
Root = fit(train_data, cutoff)
println("Cây quyết định vừa được xây dựng")
visualize(Root,-1)
y_hat = predict(Root, test_data)
print("Độ đo accuracy của tập test = ", accuracy(test_data, y_hat)*100.0, "%")