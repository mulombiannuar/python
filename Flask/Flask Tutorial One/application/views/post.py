from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from application.models.post import Post
from application import db

post = Blueprint('post', __name__)

@post.route('/')
@login_required
def list_posts():
    posts = Post.query.filter_by(user_id=current_user.id).order_by(Post.created_at.desc()).all()
    return render_template('post/posts.html', posts=posts)

@post.route('/create', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')

        if not all([title, content]):
            flash('Title and Content are required.', 'error')
            return redirect(url_for('post.create_post'))

        new_post = Post(
            title=title,
            content=content,
            user_id=current_user.id
        )

        db.session.add(new_post)
        db.session.commit()
        flash('Post created successfully!', 'success')
        return redirect(url_for('post.list_posts'))

    return render_template('post/create_post.html')

@post.route('/<int:post_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_post(post_id):
    post = Post.query.get_or_404(post_id)

    if post.user_id != current_user.id:
        flash('You are not authorized to edit this post.', 'error')
        return redirect(url_for('post.list_posts'))

    if request.method == 'POST':
        post.title = request.form.get('title')
        post.content = request.form.get('content')

        db.session.commit()
        flash('Post updated successfully!', 'success')
        return redirect(url_for('post.list_posts'))

    return render_template('post/edit_post.html', post=post)

@post.route('/<int:post_id>/delete', methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)

    if post.user_id != current_user.id:
        flash('You are not authorized to delete this post.', 'error')
        return redirect(url_for('post.list_posts'))

    db.session.delete(post)
    db.session.commit()
    flash('Post deleted successfully!', 'success')
    return redirect(url_for('post.list_posts'))
