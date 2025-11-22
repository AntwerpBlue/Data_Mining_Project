# main.py

def main():
    print("====================================")
    print("     模型运行菜单（选择一个模型）")
    print("====================================")
    print("1. 决策树模型 (Decision Tree)")
    print("2. 逻辑回归模型 (Logistic Regression)")
    print("3. SVM 模型 (SVM)")
    print("4. 随机森林模型(Random Forest)")
    print("0. 退出")
    print("====================================")

    choice = input("请输入要运行的模型编号：")

    if choice == "1":
        import Decision_Tree_model as model_lib
        model_lib.run()

    elif choice == "2":
        import Logistic_Regression_model as model_lib
        model_lib.run()

    elif choice == "3":
        import SVM_model as model_lib
        model_lib.run()

    elif choice == "4":
        import Random_Forest_model as model_lib
        model_lib.run()
    elif choice == "5":
        import KNN_model as model_lib
        model_lib.run()

    elif choice == "0":
        print("已退出程序。")
        return

    else:
        print("输入无效，请重新运行程序。")

if __name__ == "__main__":
    main()